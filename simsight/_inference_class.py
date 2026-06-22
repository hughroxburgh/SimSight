import numpy as np

from scipy import integrate
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ._compute import Transform_Points

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


def Density_To_DM(density,lengths,redshift):
    """
    Transform model density to model DM.
    """

    # -- Values -- #
    mu_H = 1.3
    mu_e = 1.167
    m_p = 1.67e-24
    
    # -- Convert to comoving free electron density -- #
    constant = 1.988e43 / (3.086e21 ** 3)   # 10^10 Msun -> g / 1kpc^3 -> cm^3
    freeElectronDensity_com = density * mu_e / (m_p * mu_H) * constant

    # -- Convert to physical free electron density -- #
    freeElectronDensity_phys = (1+redshift)**3 * freeElectronDensity_com
    
    # -- Integrate over comoving distance -- #
    DM = freeElectronDensity_phys * (lengths*1000) / (1+redshift)**2

    return DM


def Photometric_Redshifts(z_true,method,mags=None):
    from scipy.stats import norm

    z_grid = np.linspace(0,1,500)

    if method == 'simple':

        sigma = 0.028
        outlier_scale = 0.3
        outlier_fraction = 0.1

        sig_main    = sigma         * (1 + z_true)
        sig_outlier = outlier_scale * (1 + z_true)

        pdf = ((1 - outlier_fraction) * norm.pdf(z_grid, z_true, sig_main)
           +      outlier_fraction * norm.pdf(z_grid, z_true, sig_outlier))

        pdf /= np.trapz(pdf, z_grid)






class Inference:
    def __init__(self, sim, filters = ['lsst_g','lsst_r','lsst_i','lsst_z'],load_kcorrect=False):
        self.sim = sim
        self.kcorrect = None

        self.filters = filters

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
        
        return mhalo_from_mstar(galaxy_masses)


    def infer_halo_size(self, halo_mass, redshift):
        """
        R200 estimate from total halo mass, returned in comoving kpc.
        """

        rho_c = self.sim.cosmo.critical_density(redshift).to(u.Msun / u.Mpc**3)
        r200_physical = ((3 * halo_mass * u.Msun) / (4 * np.pi * 200 * rho_c)) ** (1/3)
        r200_comoving = r200_physical * (1 + redshift)
        return r200_comoving.to(u.kpc).value

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

    def infer_dm(self,subsightline,igm_background='smooth_truth',halo_params='inferred',smooth_kernel=1000):
        """
        Model the DM contribution along the grid for this sightline.
        """

        # -- Start by defining a fine grid -- #
        t_res = 10 #ckpc

        # -- Define a regular grid along sightline axis -- #
        t_grid = np.arange(t_res/2, subsightline.length, t_res)
        lengths = np.full_like(t_grid, t_res, dtype=float)
        if subsightline.length % t_res > 0:
            t_grid = np.append(t_grid, subsightline.length)          # sample at exact endpoint
            lengths = np.append(lengths, subsightline.length % t_res)

        # -- Initialise -- #
        haloAssignment = np.full(t_grid.shape[0],-1)
        cellConditions = np.full(t_grid.shape[0],0)

        # -- Define IGM_Background -- #
        if igm_background == 'smooth_truth':
            density = Resample_Sightline_Density(subsightline.sub_Grid,subsightline.sub_Density,
                                                 subsightline.sub_CellConditions == 0,t_grid,smooth_kernel)

        elif igm_background == 'mean':

            f_igm = self.sim.f_igm
            mean_density = self.sim.cosmo.Ob0 * self.sim.cosmo.critical_density0.to(1e10*u.Msun / u.kpc**3).value
            igm_density = f_igm * mean_density
            # rho_igm_background = 0.936 / 1e10  # Determined empirically from finding density above which 50% of sightline duration is spent.
            density = np.full(t_grid.shape[0],igm_density)

        #  -- Extract halo data -- #
        halo_positions = np.array([Transform_Points(subsightline,halo['Pos']) for halo in subsightline.sub_Halos if halo is not None])  
        if len(halo_positions)>0:

            if halo_params == 'inferred':
                mass_string = 'Inferred_HaloMass'
                radius_string = 'Inferred_HaloRadius'
            else:
                mass_string = 'TotalMass'
                radius_string = 'Radius'

            radii = np.array([halo[radius_string] for halo in subsightline.sub_Halos])
            ids = np.array([halo['ID'] for halo in subsightline.sub_Halos])

            # Distance from (0,0,z_mid) to halo COM
            dx2 = halo_positions[:, 0][:, None]**2
            dy2 = halo_positions[:, 1][:, None]**2
            dz2 = (halo_positions[:, 2][:, None] - t_grid[None, :])**2
            dist2 = dx2 + dy2 + dz2

            # Find grid points inside halos
            inside = dist2 <= radii[:, None]**2     # (Nhalo, Nseg)

            for j in range(inside.shape[0]):
                haloAssignment += inside[j,:].astype(int) * (ids[j]+1)

            cellConditions[np.where(haloAssignment>-1)[0]] = 1

            # -- Add modelled or true halo contribution -- #
            for j,halo in enumerate(subsightline.sub_Halos):
                mask = inside[j]

                if halo_params != False:
                    halo_density = self.halo_model(np.sqrt(dist2[j][mask]),halo[mass_string],halo[radius_string],self.sim) 
                else:
                    resampled_density = Resample_Sightline_Density(subsightline.sub_Grid,subsightline.sub_Density,
                                                                   np.ones_like(subsightline.sub_Grid).astype(bool),t_grid,0)
                    halo_density = resampled_density[mask]
                
                density[mask] = np.maximum(halo_density,density[mask])

        # -- Convert to DM -- #
        dm = Density_To_DM(density,lengths,subsightline.sub_BoxRedshifts)

        return lengths,density,dm,cellConditions,haloAssignment