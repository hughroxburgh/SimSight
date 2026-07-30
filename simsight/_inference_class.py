import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from copy import deepcopy
from scipy import integrate
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator

from ._compute import Transform_Points
from ._photz import FZBoostPredictor

def Resample_Sightline_Density(sl_grid,sl_densities,sl_halo_mask,standard_grid,smoothing_scale=1000):
    """
    Reinterpolates a non-uniform density field onto a normalised grid, and smooths with a gaussian kernel. 
    """

    # -- Find centres of each true cell -- #
    edges = np.concatenate([[0], np.cumsum(sl_grid)])
    centres = edges[:-1] + np.diff(edges)/2

    # -- Create an interpolation over only non halo cells -- #
    density_interp = interp1d(centres[sl_halo_mask], sl_densities[sl_halo_mask], 
                kind='linear', bounds_error=False,
                fill_value=(sl_densities[sl_halo_mask][0], sl_densities[sl_halo_mask][-1]))

    # -- Interpolate on standard_grid -- #
    igm_density = density_interp(standard_grid)

    # -- Apply Gaussian filter -- #
    if smoothing_scale > 0:
        sigma_cells = smoothing_scale / abs(np.diff(standard_grid)[0])
    #     igm_density = gaussian_filter1d(np.log10(igm_density), sigma=sigma_cells, mode='nearest')

    # return 10**igm_density
        igm_density = gaussian_filter1d(igm_density, sigma=sigma_cells, mode='nearest')

    return igm_density


class Inference:
    def __init__(self, sim, filters = ['lsst_g','lsst_r','lsst_i','lsst_z'],load_kcorrect=False,
                 redshift_mode='truth',kcorrect_mode='kcorrect',m2l_mode='roediger15',halomass_mode='dpowerlaw_fit',
                 halo_params='inferred',igm_background='mean',density_smooth_kernel=1000):
        
        self.sim = sim
        self.kcorrect = None

        self.filters = filters

        self.halo_inference_params = {'Redshift_Mode':redshift_mode,
                                      'KCorrect_Mode':kcorrect_mode,
                                      'M2L_Mode':m2l_mode,
                                      'HaloMass_Mode':halomass_mode}
        
        self.model_params = {'HaloParams_Mode':halo_params,
                            'IGM_Mode':igm_background}
        if igm_background == 'smooth_truth':
            self.model_params['SmoothingKernal'] = density_smooth_kernel

        if load_kcorrect:
            self._load_kcorrect()

    def _load_kcorrect(self):
        """
        Initialise the K-correction module.
        """

        print(f'Loading kcorrect with {self.filters}...',end='\r')
        import os
        os.environ["SPS_HOME"] = self.sim.fsps_path
        import fsps
        from astropy.io import ascii
        import astropy.table 
        import kcorrect

        responses_dir = os.path.join(kcorrect.KCORRECT_DIR, 'data', 'responses')
        for i, band in enumerate(self.filters, 1):
            wave_filt, trans = fsps.get_filter(band).transmission
            t = astropy.table.Table()
            t['lambda'] = wave_filt.astype('float32')
            t['pass']   = trans.astype('float32')
            ascii.write(t, os.path.join(responses_dir, f'{band}.dat'),
                        format='fixed_width', overwrite=True)

        self.kcorrect = kcorrect.kcorrect.Kcorrect(responses=list(self.filters))
        print(f'Loading kcorrect with {self.filters}...Done!')


    def infer_redshift(self,z_true=None, mags=None, mag_errs=None,
                        zmax=3.0):

        z_grid = np.linspace(0, zmax, 500)
        rng = np.random.default_rng()

        method = self.halo_inference_params['Redshift_Mode']

        if method == 'truth':
            z_phots = z_true

        if method == 'simple_phot':

            from scipy.stats import truncnorm

            sigma            = 0.028
            outlier_scale    = 0.3
            outlier_fraction = 0.1

            z_true_array = np.atleast_1d(z_true)
            is_outlier   = rng.random(len(z_true_array)) < outlier_fraction

            z_phots = np.zeros(len(z_true_array))
            for i, z in enumerate(z_true_array):
                sig = (outlier_scale if is_outlier[i] else sigma) * (1 + z)
                a   = (0 - z) / sig   # lower truncation bound
                z_phots[i] = truncnorm.rvs(a, np.inf, loc=z, scale=sig, random_state=rng)

            # # build PDFs for plotting/return consistency
            # pdfs = np.array([
            #     ((1 - outlier_fraction) * norm.pdf(z_grid, z, sigma * (1+z))
            #     +     outlier_fraction  * norm.pdf(z_grid, z, outlier_scale * (1+z)))
            #     for z in z_true_array
            # ])
            # pdfs /= np.trapezoid(pdfs, z_grid, axis=1)[:, None]

        elif method == 'flexzboost':
            assert mags is not None and mag_errs is not None, \
                "mags and mag_errs required for flexzboost"

            fzb = FZBoostPredictor.load()
            z_phots, pdfs, z_grid = fzb.predict(mags, mag_errs,3001)

        # if plot:
        #     fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        #     axes[0].plot(z_grid, pdfs[0])
        #     axes[0].set_xlabel('z')
        #     axes[0].set_ylabel('p(z)')
        #     axes[0].set_title('Example PDF (first galaxy)')

        #     axes[1].scatter(np.atleast_1d(z_true), z_phot, s=5, alpha=0.5)
        #     axes[1].plot([z_grid[0], z_grid[-1]], [z_grid[0], z_grid[-1]], 'r--', lw=1)
        #     axes[1].set_xlabel('True z')
        #     axes[1].set_ylabel('Photo-z mode')
        #     plt.tight_layout()

        return z_phots #, pdfs, z_grid


    def process_redshifts(self, sightlines, filters=None):

        method = self.halo_inference_params['Redshift_Mode']

        # -- Helper to iterate all (sightline, halo, galaxy_index) -- #
        def iter_galaxies():
            for sl in sightlines:
                for i,sh in enumerate(sl.sub_Halos):
                    if sh != [None]:
                        for halo in sh:
                            for j in range(len(halo['ObservedGalaxies'])):
                                yield halo, j

        if method == 'truth':
            for halo, j in iter_galaxies():
                halo['ObservedGalaxies'][j]['Inferred_Redshift'] = halo['Redshift']

        elif method == 'simple_phot':
            galaxy_refs = list(iter_galaxies())
            if not galaxy_refs:
                return
            true_zs     = np.array([halo['Redshift'] for halo, j in galaxy_refs])
            phot_zs     = self.infer_redshift(z_true=true_zs)

            for (halo, j), z in zip(galaxy_refs, phot_zs):
                halo['ObservedGalaxies'][j]['Inferred_Redshift'] = z

        elif method == 'flexzboost':
            assert filters is not None, "filters required for flexzboost"

            galaxy_refs = list(iter_galaxies())
            if not galaxy_refs:
                return
            mags        = np.array([
                [halo['ObservedGalaxies'][j]['ApparentMags'][band] for band in filters]
                for halo, j in galaxy_refs
            ])
            mag_errs    = np.ones_like(mags) * 0.05   # temporary

            phot_zs = self.infer_redshift(mags=mags, mag_errs=mag_errs)

            for (halo, j), z in zip(galaxy_refs, phot_zs):
                halo['ObservedGalaxies'][j]['Inferred_Redshift'] = z


    def infer_galaxy_mags(self,redshift,apparent_mags,filters=['lsst_g','lsst_r','lsst_i','lsst_z'],limiting_mags=None):
        """
        Infer galaxy absolute magnitude based on apparent mag and redshift.
        """
        
        # -- Initialise kcorrect -- #
        if filters != self.filters:
            self.filters = filters
            self._load_kcorrect()

        if limiting_mags is None:
            limiting_mags = np.array([100 for _ in range(len(self.filters))])
            
        # -- Read in magnitudes and convert to flux -- #
        # m_g,m_r,m_i,m_z = [apparent_mags[key] for key in apparent_mags.keys()]

        mag_arr = np.array([apparent_mags[key] for key in filters])
        mag_arr = np.nanmin([mag_arr,limiting_mags],axis=0)
        # mag_arr = np.array([m_g, m_r, m_i,m_z])
        maggies = [10**(-m / 2.5) for m in mag_arr]

        # -- Perform kcorrection -- #
        dm = 0.05
        ivar = [1.0 / (f * dm / 1.0857)**2 for f in maggies]
        coeffs = self.kcorrect.fit_coeffs(redshift=redshift, maggies=maggies, ivar=ivar)
        k      = self.kcorrect.kcorrect(redshift=redshift, coeffs=coeffs)

        # -- Produce k-corrected apparent magnitudes -- #
        mag_kcorr_arr = mag_arr - k

        # -- Produce k-corrected absolute magnitudes
        D_L = self.sim.cosmo.luminosity_distance(redshift).to('pc').value
        mu  = 5 * np.log10(D_L / 10)
        Mag_kcorr_arr = mag_kcorr_arr - np.ones_like(mag_arr) * mu
        
        Mag_kcorr_dict = {filters[i]: Mag_kcorr_arr[i] for i in range(len(filters))}

        return Mag_kcorr_dict

    def infer_galaxy_mass(self, mags):
        """
        Infer galaxy mass based on colour.
        """

        mag1 = mags['lsst_g']
        mag2 = mags['lsst_i']

        colour = mag1 - mag2

        log_mass_over_lum = 1.231 * colour - 0.811  # log(M/L_g) in solar units

        M_sun_g = 5.12
        L_g = 10**((M_sun_g - mag1) / 2.5)  # L_sun

        stellar_mass = 10**log_mass_over_lum * L_g  # M_sun

        return stellar_mass

    def infer_halo_mass(self,galaxy_masses):
        """
        Infer total halo mass based on sum of stellar masses in halo.

        Currently based on a redshift independent double power law model fit to TNG 100-3 data: temporary!
        """
        
        if len(galaxy_masses) == 0:
            return np.nan

        total_stellar_mass = np.nansum([m for m in galaxy_masses if m is not None])

        def moster_model(halo_mass):
            N = 9.29081662e-03
            log_max_mass = 12.076
            beta = 1.443
            gamma = 0.325
            
            max_mass = 10**log_max_mass
            galmass = 2 * N * halo_mass * ((halo_mass / max_mass)**(-beta) + (halo_mass / max_mass)**(gamma))**-1
            return galmass

        mhalo_grid = np.logspace(10, 16, 10000)
        mstar_grid = moster_model(mhalo_grid)

        mhalo_from_mstar = interp1d(mstar_grid, mhalo_grid, 
                                    bounds_error=False, 
                                    fill_value=np.nan)
        
        return float(mhalo_from_mstar(total_stellar_mass))


    def infer_halo_size(self, halo_mass, redshift):
        """
        R200 estimate from total halo mass, returned in comoving kpc.
        """

        rho_c = self.sim.cosmo.critical_density(redshift).to(u.Msun / u.Mpc**3)
        r200_physical = ((3 * halo_mass * u.Msun) / (4 * np.pi * 200 * rho_c)) ** (1/3)
        r200_comoving = r200_physical * (1 + redshift)
        return r200_comoving.to(u.kpc).value


    # ------------------------ Model DM Contribution ------------------------ #

    def halo_model(self,radii, halo_m200,halo_r200, sim,f_gas=0.75, alpha=2.0, y0=2.0):
        """
        mNFW halo model for baryonic density. 
        """
        
        # -- Define import values -- #
        omega_b = sim.cosmo.Ob0     # Baryon Density
        omega_m = sim.cosmo.Om0     # Total Matter Density
        c200 = 4.67 * (halo_m200 * self.sim.hub / 1e14) ** (-0.11)      # Halo concentration parameter

        # -- Estimate CGM Mass and thus rho_0 (normalisation factor) -- #
        M_cgm = f_gas * (omega_b / omega_m) * halo_m200
        norm_integrand = lambda y: y ** (1.0 + alpha) / (y0 + y) ** (2.0 + alpha)
        norm_integral, _ = integrate.quad(norm_integrand, 0.0, c200)
        rho0 = M_cgm / (4.0 * np.pi * (halo_r200 / c200) ** 3 * norm_integral)

        # -- Evaluate Profile -- #
        r = np.atleast_1d(np.asarray(radii, dtype=float))
        y = c200 * r / halo_r200
        y = np.maximum(y, 1e-10)
        rho_b = rho0 / (y ** (1.0 - alpha) * (y0 + y) ** (2.0 + alpha))

        return rho_b/1e10


    def _fit_and_rescale_smooth_igm_density(self,sightlines, deg=1, inplace=True):
        """
        1. Computes length-weighted mean smoothed IGM density per redshift snapshot.
        2. Fits a polynomial mean_density(z) through those points.
        3. Rescales sub_DensityIGM to cosmic_mean * (1 + delta),
        delta = sub_DensityIGM / mean_density(z_segment) - 1.

        Parameters
        ----------
        inplace : bool
            If True, modifies the given sightlines directly and returns them.
            If False, deep-copies each sightline before rescaling, leaving the
            originals untouched.

        Returns
        -------
        mean_density : callable, fitted mean_density(z)
        z_arr, mean_arr, coeffs : diagnostic fit info
        rescaled_sightlines : the (in-place or deep-copied) rescaled sightlines
        """
        redshifts = np.unique(sightlines[0].sub_BoxRedshifts[:sightlines[0].subsightline_reached(halos=True)])

        z_arr, mean_arr, weight_arr = [], [], []
        cosmic_mean = self.sim.cosmo.Ob0 * self.sim.cosmo.critical_density0.to(
                1e10 * u.Msun / u.kpc**3).value

        for z in tqdm(redshifts,desc='    calculating mean_IGM(z)'):
            all_smoothed, all_lengths = [], []
            for sl in sightlines:
                for j, ssl in enumerate(sl.modelled.sub_DensityIGM):
                    if sl.sub_BoxRedshifts[j] == z:
                        all_smoothed.append(ssl)
                        all_lengths.append(sl.modelled.sub_Grid[j])

            if len(all_smoothed) == 0:
                continue

            lengths_concat = np.concatenate(all_lengths)
            z_arr.append(z)
            mean_arr.append(np.average(np.concatenate(all_smoothed), weights=lengths_concat))
            weight_arr.append(len(lengths_concat))

        z_arr = np.array(z_arr)
        mean_arr = np.array(mean_arr)
        weight_arr = np.array(weight_arr)

        coeffs = np.polyfit(z_arr, mean_arr, deg=deg, w=weight_arr)

        def mean_density_model(z):
            return np.polyval(coeffs, np.atleast_1d(z))

        # -- rescale each sightline (in place or via deepcopy) -- #
        rescaled_sightlines = []
        for sl in tqdm(sightlines, desc='    rescaling sub_DensityIGM'):
            sl_rescaled = sl if inplace else deepcopy(sl)
            for j, ssl in enumerate(sl_rescaled.modelled.sub_DensityIGM):
                z_seg = sl_rescaled.sub_BoxRedshifts[j]
                local_mean = mean_density_model(z_seg)[0]
                delta = ssl / local_mean - 1
                sl_rescaled.modelled.sub_DensityIGM[j] = cosmic_mean * (1 + delta)
            rescaled_sightlines.append(sl_rescaled)

        return mean_density_model, z_arr, mean_arr, coeffs, rescaled_sightlines


    def model_dm_partition(self,subsightline):
        
        """
        Run once per sightline, OUTSIDE the MCMC loop. Computes the unit-scale
        (f_gas=1, f_igm=1) density contributions from IGM and halo components
        separately, on the fine grid, without combining them. This is the
        expensive step (mNFW integration, IGM resampling) and should never be
        re-run per MCMC step.
        """
        t_res = 10  # ckpc
        t_grid = np.arange(t_res / 2, subsightline.length, t_res)
        lengths = np.full_like(t_grid, t_res, dtype=float)
        if subsightline.length % t_res > 0:
            t_grid = np.append(t_grid, subsightline.length)
            lengths = np.append(lengths, subsightline.length % t_res)

        # -- IGM unit density (f_igm = 1) -- #
        if self.model_params['IGM_Mode'] == 'smooth_truth':
            if len(np.where(subsightline.sub_CellConditions == 0)[0]) == 0:
                mean_density = self.sim.cosmo.Ob0 * self.sim.cosmo.critical_density0.to(
                                1e10 * u.Msun / u.kpc**3).value
                density_igm_unit = np.full(t_grid.shape[0], mean_density)
            else:
                density_igm_unit = Resample_Sightline_Density(
                    subsightline.sub_Grid, subsightline.sub_Density,
                    subsightline.sub_CellConditions == 0, t_grid,
                    self.model_params['SmoothingKernal']
                )
        elif self.model_params['IGM_Mode'] == 'mean':
            mean_density = self.sim.cosmo.Ob0 * self.sim.cosmo.critical_density0.to(
                1e10 * u.Msun / u.kpc**3).value
            density_igm_unit = np.full(t_grid.shape[0], mean_density)

        # -- Halo unit density (f_gas = 1), per-halo, kept separate -- #
        density_halo_unit = np.zeros_like(t_grid)
        halo_assignment = np.full(t_grid.shape[0], -1)
        cell_conditions = np.zeros(t_grid.shape[0], dtype=int)

        halo_positions = np.array([
            Transform_Points(subsightline, h['Pos'])
            for h in subsightline.sub_Halos if h is not None
        ])

        if len(halo_positions) > 0:
            radii = np.array([h['Radius'] for h in subsightline.sub_Halos])
            ids = np.array([h['ID'] for h in subsightline.sub_Halos])

            dx2 = halo_positions[:, 0][:, None] ** 2
            dy2 = halo_positions[:, 1][:, None] ** 2
            dz2 = (halo_positions[:, 2][:, None] - t_grid[None, :]) ** 2
            dist2 = dx2 + dy2 + dz2
            inside = dist2 <= radii[:, None] ** 2

            for j in range(inside.shape[0]):
                halo_assignment += inside[j, :].astype(int) * (ids[j] + 1)
            cell_conditions[halo_assignment > -1] = 1

            for j, halo in enumerate(subsightline.sub_Halos):
                mask = inside[j]
                if self.model_params['HaloParams_Mode'] != 'off':
                    # f_gas fixed to 1 here -- unit profile, rescaled later
                    unit_density_j = self.halo_model(
                        np.sqrt(dist2[j][mask]), halo['TotalMass'],
                        halo['Radius'], self.sim, f_gas=1.0
                    )
                else:
                    resampled = Resample_Sightline_Density(
                        subsightline.sub_Grid, subsightline.sub_Density,
                        np.ones_like(subsightline.sub_Grid).astype(bool),
                        t_grid, 0
                    )
                    density_true_j = resampled[mask]

                    m_gas_true_h = halo['GasMass']

                    if m_gas_true_h <= 0:
                        # no true gas recorded for this halo -- renormalization is undefined.
                        # Treat as contributing zero to the reference density (the halo held
                        # no gas to renormalize), rather than blowing up.
                        unit_density_j = np.zeros_like(density_true_j)
                    else:
                        omega_b = self.sim.cosmo.Ob0
                        omega_m = self.sim.cosmo.Om0
                        scale_h = (omega_b / omega_m) * halo['TotalMass'] / m_gas_true_h
                        unit_density_j = density_true_j * scale_h

                # still take elementwise max ACROSS OVERLAPPING HALOS
                # (this max is fine -- it's halo-vs-halo, not halo-vs-IGM,
                # and doesn't depend on f_gas/f_igm since both use f_gas=1
                # here uniformly)
                density_halo_unit[mask] = np.maximum(density_halo_unit[mask], unit_density_j)

        return lengths, density_igm_unit, density_halo_unit, cell_conditions, halo_assignment
    



    def build_halo_arrays(self, sightlines, redshift):
        """
        Flatten per-halo unit CGM DM contributions across all sightlines.

        Returns:
        halo_logM    : log10(halo mass [Msun]) for every halo intersection
        dm_halo_unit : modelled, fgas=1-equivalent DM contribution per halo
                        (i.e. whatever halo_info returns under 'Compute' when
                        modelled=True -- assumed to already be the unit/fgas=1
                        value; scale by f_gas(M) afterward in the likelihood)
        sl_index     : sightline index (0..n_sightlines-1) for each entry
        n_sightlines : total number of sightlines
        """
        halo_logM_list = []
        dm_unit_list = []
        sl_index_list = []

        for i, sl in enumerate(tqdm(sightlines, desc='    building per-halo unit CGM arrays')):
            halos = sl.halo_info(with_compute=True, modelled=True, redshift=redshift,fgas=1)

            if len(halos) == 0:
                continue

            masses = np.array([h['TotalMass'] for h in halos.values()])
            dms = np.array([h['Compute'] for h in halos.values()])

            good = masses > 0
            if not np.any(good):
                continue

            halo_logM_list.append(np.log10(masses[good]))
            dm_unit_list.append(dms[good])
            sl_index_list.append(np.full(good.sum(), i))

        halo_logM = np.concatenate(halo_logM_list)
        dm_halo_unit = np.concatenate(dm_unit_list)
        sl_index = np.concatenate(sl_index_list)

        return halo_logM, dm_halo_unit, sl_index, len(sightlines)


    def fgas_of_mass(self, logM, anchor_logM, fgas_anchors):
        """
        Piecewise-linear f_gas(M) through free anchor y-values, with linear
        extrapolation beyond the outermost anchors.
        """
        interp = interp1d(anchor_logM, fgas_anchors, kind='linear',
                        fill_value='extrapolate', assume_sorted=True)
        vals = interp(logM)
        return np.clip(vals, 0.0, 1.0)


    def build_sigma_igm(self, sightlines, cosmo, redshift, mode='log', dm_floor=1e-4):
        z_val = np.atleast_1d(redshift)
        dm_igm_true = np.array([
            s.extract_compute(cosmo, redshift=z_val, environment='IGM', modelled=False)
            for s in tqdm(sightlines, desc='    building sigma_igm (truth)')
        ])
        dm_igm_model = np.array([
            s.extract_compute(cosmo, redshift=z_val, environment='IGM', modelled=True,
                            figm=self.sim.figm)
            for s in tqdm(sightlines, desc='    building sigma_igm (model)')
        ])

        if mode == 'log':
            resid = np.log10(np.maximum(dm_igm_true, dm_floor)) - \
                    np.log10(np.maximum(dm_igm_model, dm_floor))
        elif mode == 'linear':
            resid = dm_igm_true - dm_igm_model

        return np.std(resid, axis=0)[0]


    def build_sigma_halo(self, sightlines, cosmo, redshift, mode='log',dm_floor=1e-4):
        z_vals = np.atleast_1d(redshift)
        dm_halo_true = np.array([
            s.extract_compute(cosmo, redshift=z_vals, environment='CGM', modelled=False)
            for s in tqdm(sightlines, desc='    building sigma_halo (truth)')
        ])
        dm_halo_model = np.array([
            s.extract_compute(cosmo, redshift=z_vals, environment='CGM', modelled=True,
                            fgas=self.sim.fgas)
            for s in tqdm(sightlines, desc='    building sigma_halo (model)')
        ])
        if mode == 'log':
            resid = np.log10(np.maximum(dm_halo_true, dm_floor)) - \
                    np.log10(np.maximum(dm_halo_model, dm_floor))
        elif mode == 'linear':
            resid = dm_halo_true - dm_halo_model

        return np.std(resid, axis=0)[0]

    

    def log_likelihood(self, theta, anchor_logM, halo_logM, dm_halo_unit, sl_index,
                    n_sightlines, dm_igm_unit, y_true,
                    sigma, mode='log',dm_floor=1e-4):
        n_anchors = len(anchor_logM)
        fgas_anchors = theta[:n_anchors]
        f_igm = theta[n_anchors]

        fgas_per_halo = self.fgas_of_mass(halo_logM, anchor_logM, fgas_anchors)
        scaled_halo_dm = fgas_per_halo * dm_halo_unit
        dm_halo_total = np.zeros(n_sightlines)
        np.add.at(dm_halo_total, sl_index, scaled_halo_dm)

        model_dm = dm_halo_total + f_igm * dm_igm_unit
        
        if mode == 'log':
            model_dm_floored = np.maximum(model_dm, dm_floor)
            y_true_floored = np.maximum(y_true, dm_floor)
            resid = np.log10(y_true_floored) - np.log10(model_dm_floored)
        elif mode == 'linear':
            resid = y_true - model_dm

        return np.sum(-0.5 * (resid**2 / sigma**2 + np.log(2 * np.pi * sigma**2)))
    

    def log_prior(self, theta, priors):
        for val, (lo, hi) in zip(theta, priors.values()):
            if not (lo < val < hi):
                return -np.inf
        return 0.0

    def log_probability(self, theta, anchor_logM, halo_logM, dm_halo_unit, sl_index,
                         n_sightlines, dm_igm_unit, y_true, sigma, priors,mode):
        lp = self.log_prior(theta, priors)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, anchor_logM, halo_logM, dm_halo_unit,
                                         sl_index, n_sightlines, dm_igm_unit, y_true, sigma,mode)