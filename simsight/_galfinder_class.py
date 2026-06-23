import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.cluster import DBSCAN
from time import time as clock
from matplotlib.patches import Ellipse
from shapely.geometry import Point
from shapely.affinity import scale, rotate
from collections import defaultdict
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator


from ._compute import Transform_Points
from ._utils import _Progress_Print

def load_grids(path,redshift):

    metallicity_grid = f'{path}/metallicity_grid.npy'
    ages_grid = f'{path}/ages_grid.npy'
    wavelength_grid = f'{path}/wavelength_grid.npy'
    mass_frac_grid = f'{path}/mass_fraction_grid.npy'
    spec_path = f'{path}/fsps_spec_grid_z{redshift:.3f}.npy'
    cf_path = f'{path}/cf_grid.npy'

    return (np.load(spec_path), np.load(wavelength_grid),
            np.load(metallicity_grid), np.load(ages_grid),
            np.load(mass_frac_grid), np.load(cf_path))


class GalaxyFinder():

    def __init__(self, snapshot, sim, stars=None, gas=False, start_sort=True, nstars_limit=3):

        self.sim = sim
        self.snapshot = snapshot
        self.stars = stars
        self.gas = gas
        self.nstars_limit = nstars_limit

        self.gather_data()

        if start_sort:
            self.sort_data()

    def gather_data(self):

        # ensure halo cache is loaded for this snapshot
        cache = self.sim._halo_cache
        
        needs_reload = (cache is None or cache['snapshot'] != self.sim._get_snap_num(self.snapshot))
        needs_stars = self.stars is not False and (cache is None or 'star_ids' not in cache)
        needs_gas = self.gas is not False and (cache is None or 'gas_ids' not in cache)

        if needs_reload or needs_stars or needs_gas:
            self.sim.load_halos(self.sim._get_snap_num(self.snapshot), return_dict=False,
                                cache_stars_particles=self.stars is not False,
                                cache_gas_particles=self.gas is not False)

        if self.stars is not False:
            self.stars = self.sim.load_data('stars', ['Coordinates', 'Masses','Metallicity','StellarFormationTime'], 
                                            self.sim._get_snap_num(self.snapshot)) if self.stars is None else self.stars
            
            z_form = (1.0 / self.stars['StellarFormationTime']) - 1.0
            t_snap = self.sim.cosmo.age(self.sim.redshifts[self.snapshot]).to(u.Gyr).value
            t_form = self.sim.cosmo.age(z_form).to(u.Gyr).value
            self.stars['Ages'] = t_snap - t_form
            del(self.stars['StellarFormationTime'])
            
            if 'ParticleIDs' not in self.stars:
                self.stars['ParticleIDs'] = self.sim._halo_cache['star_ids']

        if self.gas is not False:
            self.gas = self.sim.load_data('gas', ['Coordinates', 'Masses'], self.sim._get_snap_num(self.snapshot)) if self.gas is None else self.gas
            if 'ParticleIDs' not in self.gas:
                self.gas['ParticleIDs'] = self.sim._halo_cache['gas_ids']

    def sort_data(self):
        
        msg = f'    sorting data'
        ts = clock()
        print(msg,end='\r')

        if self.stars is not False:
            sorted_order = np.argsort(self.stars['ParticleIDs'])
            self.stars = {key: self.stars[key][sorted_order] for key in self.stars.keys()}

        if self.gas is not False:
            sorted_order = np.argsort(self.gas['ParticleIDs'])
            self.gas = {key: self.gas[key][sorted_order] for key in self.gas.keys()}
        
        _Progress_Print(msg,ts)

    def load_halo(self,halo_id,verbose=True):
        """
        Requires stars and gas be sorted.
        """

        
        if verbose:
            msg = f'Loading halo info'
            ts = clock()
            print(msg,end='\r')
        halo_info = self.sim.load_halo(self.sim._get_snap_num(self.snapshot),halo_id,
                                       load_stars= self.stars is not False,
                                       load_gas= self.gas is not False)
        if verbose:
            _Progress_Print(msg,ts)


        stars = None
        gas = None
        if self.stars is not False:
            if verbose:
                msg = f'Loading stars'
                ts = clock()
                print(msg,end='\r')
            pos = np.searchsorted(self.stars['ParticleIDs'], halo_info['StarsParticleIDs'])
            pos = pos.clip(0, len(self.stars['ParticleIDs']) - 1)
            mask = self.stars['ParticleIDs'][pos] == halo_info['StarsParticleIDs']
            idx = pos[mask]
            stars = {key: self.stars[key][idx] for key in self.stars.keys()}
            if verbose:
                _Progress_Print(msg,ts)

        if self.gas is not False:
            if verbose:
                msg = f'Loading gas'
                ts = clock()
                print(msg,end='\r')
            pos = np.searchsorted(self.gas['ParticleIDs'], halo_info['GasParticleIDs'])
            pos = pos.clip(0, len(self.gas['ParticleIDs']) - 1)
            mask = self.gas['ParticleIDs'][pos] == halo_info['GasParticleIDs']
            idx = pos[mask]
            gas = {key: self.gas[key][idx] for key in self.gas.keys()}
            if verbose:
                _Progress_Print(msg,ts)
        
        if verbose:
            print('\n')

        return halo_info, stars, gas 

    
    def cluster_stars(self, halo_id, alpha=0.86, linking_length=20, min_stars=10, mass_frac=0.9, sig_frac=0.2, stars=None, plot=True,verbose=True):

        halo_info,stars,_ = self.load_halo(halo_id,verbose)

        linking_length_comoving = linking_length * (1 + self.sim.redshifts[self.snapshot])**(-alpha)

        box_size = self.sim.box_size  # in ckpc

        delta = stars['Coordinates'] - halo_info['Pos']
        delta[delta > box_size / 2] -= box_size
        delta[delta < -box_size / 2] += box_size

        stars['Coordinates'] = halo_info['Pos'] + delta

        cluster = DBSCAN(eps=linking_length_comoving, min_samples=min_stars).fit(stars['Coordinates'])
        labels = cluster.labels_.copy()

        unique_clusters = np.unique(labels[labels != -1])

        # single cluster or no clusters - treat everything as one object
        if len(unique_clusters) <= 1:
            labels = np.zeros(len(labels), dtype=int)
            unique_clusters = np.array([0])
            cluster_centres = np.array([np.average(stars['Coordinates'], weights=stars['Masses'], axis=0)])
            significant = np.array([0])

        else:
            # compute cluster centres in 3D
            cluster_centres = np.array([
                np.average(stars['Coordinates'][labels == c], weights=stars['Masses'][labels == c], axis=0)
                for c in unique_clusters
            ])

            # iteratively expand linking length to absorb noise until mass_frac is reached
            noise_mask = labels == -1
            noise_positions = stars['Coordinates'][noise_mask]
            dists = np.linalg.norm(noise_positions[:, None, :] - cluster_centres[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            nearest = unique_clusters[np.argmin(dists, axis=1)]

            count = 0
            fraction = np.nansum(stars['Masses'][labels != -1]) / np.nansum(stars['Masses'])
            while fraction < mass_frac and count < 100:
                wider_linking = linking_length_comoving * (3 + count)
                labels[noise_mask] = np.where(min_dists < wider_linking, nearest, -1)
                fraction = np.nansum(stars['Masses'][labels != -1]) / np.nansum(stars['Masses'])
                count += 1

            # apply significance threshold
            cluster_masses = np.array([np.nansum(stars['Masses'][labels == c]) for c in unique_clusters])
            max_mass = cluster_masses.max()
            significant = unique_clusters[cluster_masses > sig_frac * max_mass]

        if plot:
            fraction = np.nansum(stars['Masses'][labels != -1]) / np.nansum(stars['Masses'])
            print(f'Linking length : {linking_length_comoving:.1f} ckpc')
            print(f'Mass in clusters = {fraction * 100:.1f}%')
            print(f'Significant clusters: {len(significant)}')

            cmap = plt.cm.get_cmap('tab10', max(unique_clusters.max() + 1, 1))
            colors = np.array([cmap(label) if label != -1 else (0, 0, 0, 0.3) for label in labels])

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            projections = [(0, 1, 'X [ckpc]', 'Y [ckpc]'),
                        (0, 2, 'X [ckpc]', 'Z [ckpc]'),
                        (1, 2, 'Y [ckpc]', 'Z [ckpc]')]

            for ax, (i, j, xlabel, ylabel) in zip(axes, projections):
                ax.scatter(stars['Coordinates'][:, i], stars['Coordinates'][:, j], s=1, c=colors)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_aspect('equal')
                ax.scatter(cluster_centres[significant][:, i], cluster_centres[significant][:, j],
                        c='k',edgecolor='k', marker='*', s=50, zorder=10,lw=0.5)

            plt.tight_layout()


        mask = np.isin(labels, significant)
        labels[~mask] = -1
        _, newlabels = np.unique(labels[mask], return_inverse=True)
        labels[mask] = newlabels

        stars['ClusterIDs'] = labels

        return halo_info,stars,cluster_centres[significant]
    

    def resolve_clusters(self,sightline,redshift,stars,centres,lsst_fwhm=0.7,merge_threshold = 0.3,plot=True,verbose=True):

        D_A = self.sim.cosmo.angular_diameter_distance(redshift).to('kpc').value

        coords = Transform_Points(sightline,stars['Coordinates'])[:,:2]

        coords[:, 0] = (coords[:, 0] / (1 + redshift)) / D_A * 206265
        coords[:, 1] = (coords[:, 1] / (1 + redshift)) / D_A * 206265

        tCentres = Transform_Points(sightline,centres)[:,:2]

        tCentres[:, 0] = (tCentres[:, 0] / (1 + redshift)) / D_A * 206265
        tCentres[:, 1] = (tCentres[:, 1] / (1 + redshift)) / D_A * 206265

        valid_mask = stars['ClusterIDs'] > -1
        valid_ids = np.unique(stars['ClusterIDs'][valid_mask])
        n = len(valid_ids)

        shapely_ellipses = []
        ellipse_metadata = []

        def get_lsst_ellipse_params(cluster_coords, cluster_masses, psf_fwhm=0.7):

            ctr = np.average(cluster_coords, weights=cluster_masses, axis=0)
            centered = cluster_coords - ctr
            cov = np.cov(centered.T, aweights=cluster_masses) # orientation via Covariance Matrix
            vals, vecs = np.linalg.eigh(cov)
            
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]    # Sort eigenvalues (descending)
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))   
            
            dists = np.sqrt(np.nansum(centered**2, axis=1))    # scale to R90
            r90 = np.percentile(dists, 90)
            
            width = 2 * r90 * np.sqrt(vals[0] / np.mean(vals))   # Calculate physical width/height
            height = 2 * r90 * np.sqrt(vals[1] / np.mean(vals))
            
            psf_buffer = psf_fwhm / 2.355  # LSST PSF Buffer (FWHM to Sigma)
            
            return ctr, width, height, theta, psf_buffer

        
        # --- Step 1: Generate PSF-convolved Ellipses --- #
        for c_id in valid_ids:
            cluster_mask = stars['ClusterIDs'] == c_id

            centre, width, height, theta, psf_buffer = get_lsst_ellipse_params(coords[cluster_mask],stars['Masses'][cluster_mask],lsst_fwhm)
                    
            # Create the physical R90 ellipse
            el = Point(centre).buffer(0.5) 
            el = scale(el, xfact=width, yfact=height)
            el = rotate(el, theta)
            
            # Apply the LSST resolution "blur"
            el_lsst = el.buffer(psf_buffer)
            
            shapely_ellipses.append(el_lsst)
            ellipse_metadata.append({'id': c_id, 'ctr': centre, 'w': width, 'h': height, 'ang': theta, 'psf_buffer': psf_buffer})


        # --- Step 2: Merge Logic (Union-Find) ---
        parent = {i: i for i in range(n)}
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j: parent[root_i] = root_j

        # Intersection over Minimum Area (Better for Minor Mergers/Satellites)
        if verbose:
            print(f"Checking overlaps (Threshold > {merge_threshold})...")

        for i in range(n):
            for j in range(i+1, n):
                inter_area = shapely_ellipses[i].intersection(shapely_ellipses[j]).area
                min_area = min(shapely_ellipses[i].area, shapely_ellipses[j].area)
                
                overlap_coeff = inter_area / min_area if min_area > 0 else 0
                
                if overlap_coeff > merge_threshold:
                    if verbose:
                        print(f"Checking overlaps (Threshold > {merge_threshold})...")
                        print(f"  [MERGE] Cluster {valid_ids[i]} & {valid_ids[j]} | Overlap: {overlap_coeff:.2f}")
                    union(i, j)

        # --- Step 3: Final Galaxy Aggregation & NEW Ellipse Calculation ---
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(valid_ids[i])

        labels = stars['ClusterIDs'].copy()

        for g_idx, cluster_list in enumerate(groups.values()):
            # 1. Mask all stars belonging to this merged group
            mask = np.isin(labels, cluster_list)
            labels[mask] = g_idx

        unique_clusters = np.unique(labels[labels>-1])
        
        if plot:
            cmap = plt.cm.get_cmap('tab10', max(unique_clusters.max() + 1, 1))
            colors = np.array([cmap(label) if label != -1 else (0, 0, 0, 0.3) for label in labels])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(coords[:, 0], coords[:, 1], s=1, c='k',alpha=0.2)

            for i,c_id in enumerate(unique_clusters):
                cluster_mask = labels == c_id
                centre, width, height, theta, psf_buffer = get_lsst_ellipse_params(coords[cluster_mask],stars['Masses'][cluster_mask],lsst_fwhm)
                color = cmap(i)
                phys_ell = Ellipse(xy=centre, width=width, height=height, 
                                angle=theta, edgecolor=color, facecolor='none', 
                                lw=2, alpha=1.0, zorder=5,
                                label=f"{c_id}")

                # LSST PSF Blur (The 'Observed' footprint)
                blur_ell = Ellipse(xy=centre, width=width + (2 * psf_buffer), 
                                height=height + (2 * psf_buffer), 
                                angle=theta, edgecolor=color, facecolor=color, 
                                alpha=0.15, linestyle='--', zorder=4)
                ax.add_patch(phys_ell)
                ax.add_patch(blur_ell)

            ax.set_title(f"Resolved Galaxies")
            ax.set_xlabel('X ["]')
            ax.set_ylabel('Y ["]')
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.show()

        stars['ClusterIDs'] = labels

        return stars,tCentres
    
    def extract_clusters(self, stars, max_radius=30):

        max_radius_phys = max_radius * (1 + self.sim.redshifts[self.snapshot])

        valid_mask = stars['ClusterIDs'] > -1
        valid_ids = np.unique(stars['ClusterIDs'][valid_mask])

        clusters = []
        masses = []
        for c_id in valid_ids:
            cluster_mask = stars['ClusterIDs'] == c_id
            cluster_coords = stars['Coordinates'][cluster_mask]
            
            if len(cluster_coords) == 0:
                continue

            centroid = np.median(cluster_coords, axis=0)
            
            distances = np.linalg.norm(cluster_coords - centroid, axis=1)
            radius_mask = distances < max_radius_phys
            
            if not np.any(radius_mask):
                continue

            cluster_stars = {key: stars[key][cluster_mask][radius_mask] for key in stars.keys()}
            clusters.append(cluster_stars)
            masses.append(cluster_stars['Masses'].sum() * 1e10)

        return clusters,masses
    

    def cluster_mags(self, cluster, redshift, grid_path, filters=['lsst_g','lsst_r','lsst_i','lsst_z'], apply_dust=True):

        import os
        os.environ["SPS_HOME"] = self.sim.fsps_path
        import fsps

        # -- Load FSPS and Dust Attenuation Grids -- #
        spectra_grid, wavelength, metallicity_grid, age_grid, mass_frac_grid, dust_grid = load_grids(grid_path, self.sim.redshifts[self.snapshot])

        # -- Extract star properties -- #
        log_solar_metallicities = np.log10(cluster['Metallicity'] / 0.0127)
        ages = cluster['Ages']

        # -- Clamp star properties to grid limits -- #
        log_metallicities = np.clip(log_solar_metallicities, metallicity_grid[0], metallicity_grid[-1])
        log_ages = np.clip(np.log10(ages), np.log10(age_grid[0]), np.log10(age_grid[-1]))

        query_pts = np.column_stack([log_metallicities, log_ages])

        log_tage_axis = np.log10(age_grid)

        # -- Recover initial masses from surviving mass fraction -- #
        mass_frac_interp = RegularGridInterpolator(
            (metallicity_grid, log_tage_axis), mass_frac_grid,
            method='linear', bounds_error=False, fill_value=None
        )
        surviving_fractions = mass_frac_interp(query_pts)
        initial_masses = (cluster['Masses'] * 1e10) / surviving_fractions

        # -- Interpolate spectra at star properties -- #
        spectra_interp = RegularGridInterpolator(
            (metallicity_grid, log_tage_axis), spectra_grid,
            method='linear', bounds_error=False, fill_value=None
        )
        spectra = spectra_interp(query_pts)

        # -- Apply Charlot & Fall attenuation per particle -- #
        if apply_dust:
            cf_interp = RegularGridInterpolator((log_tage_axis,), dust_grid, method='linear', bounds_error=False, fill_value=None)
            atten = cf_interp(log_ages[:, None])
            spectra = spectra * atten

        # -- Scale by initial mass and sum over particles -- #
        L_nu = np.nansum(initial_masses[:, None] * spectra, axis=0)         # Luminosity density, units: Lsun/Hz 

        # -- Calculate the absolute magnitudes using the rest wavelength or the observed wavelength -- #
        absolute_mags_obs = {}
        absolute_mags_rest = {}
        for band in filters:
            wave_filt, trans = fsps.get_filter(band).transmission

            # Transmission in rest frame or observed frame 
            T_rest = np.interp(wavelength, wave_filt, trans, left=0.0, right=0.0)   
            T_obs = np.interp(wavelength, wave_filt / (1 + redshift), trans, left=0.0, right=0.0)   

            # Produce band averaged L_nu
            for T, store, z in [(T_obs, absolute_mags_obs, redshift), (T_rest, absolute_mags_rest, 0)]:
                num = np.trapz(L_nu * T / wavelength**2, wavelength)
                den = np.trapz(T / wavelength**2, wavelength)

                if den <= 0:
                    store[band] = np.nan
                    continue

                L_nu_bandavg = float((num / den) * (1+z))  # incorporates the boosting from the scrunching of frequency 

                # Convert to AB magnitude 
                store[band] = -2.5 * np.log10(L_nu_bandavg) - 32.36 if L_nu_bandavg > 0 else np.nan   # Derived from AB magnitude zeropoint
        
        cluster['AbsoluteMags'] = absolute_mags_rest
        # cluster['AbsoluteMagsObs'] = absolute_mags_obs

        # -- Calculate apparent magnitudes using the distance modulus and luminosity distance -- #
        D_L = self.sim.cosmo.luminosity_distance(redshift).to('pc').value
        distance_modulus = 5 * np.log10(D_L / 10)  # Convert Mpc to pc for distance modulus

        apparent_mags = {band: mag + distance_modulus if not np.isnan(mag) else np.nan for band, mag in absolute_mags_obs.items()}

        cluster['ApparentMags'] = apparent_mags

        return cluster


        
        # # --
        # absolute_mags_obs = {}
        # for band, T in filters_on_wave_obs.items():

        #     num = np.trapz(L_nu * T / wavelength**2, wavelength)
        #     den = np.trapz(T / wavelength**2, wavelength)

        #     if den <= 0:
        #         absolute_mags_obs[band] = np.nan
        #         continue

        #     L_nu_bandavg = float((num / den) * (1 + redshift))

        #     if L_nu_bandavg <= 0:
        #         absolute_mags_obs[band] = np.nan
        #     else:
        #         absolute_mags_obs[band] = -2.5 * np.log10(L_nu_bandavg) - 32.3677

        # cluster['AbsoluteMagsObs'] = absolute_mags_obs

        # # -- Convolve with filters and compute AB absolute magnitudes -- #
        # absolute_mags_rest = {}
        # for band, T in filters_on_wave_rest.items():

        #     num = np.trapz(L_nu * T / wavelength**2, wavelength)
        #     den = np.trapz(T / wavelength**2, wavelength)

        #     if den <= 0:
        #         absolute_mags_rest[band] = np.nan
        #         continue

        #     L_nu_bandavg = float(num / den)

        #     if L_nu_bandavg <= 0:
        #         absolute_mags_rest[band] = np.nan
        #     else:
        #         absolute_mags_rest[band] = -2.5 * np.log10(L_nu_bandavg) - 32.3677

        # cluster['AbsoluteMagsRest'] = absolute_mags_rest

        # return cluster

    # def cluster_apparent_mags(self,cluster,redshift,grid_path,filters=['lsst_g','lsst_r','lsst_i'],apply_dust=True):

    #     cluster = self.cluster_absolute_mags(cluster,redshift,grid_path,filters,apply_dust)

    #     D_L = self.sim.cosmo.luminosity_distance(redshift).to('pc').value
    #     distance_modulus = 5 * np.log10(D_L / 10)  # Convert Mpc to pc for distance modulus

    #     apparent_mags = {band: mag + distance_modulus if not np.isnan(mag) else np.nan for band, mag in cluster['AbsoluteMagsObs'].items()}

    #     cluster['ApparentMags'] = apparent_mags

    #     return cluster
    
    def process_halo(self,halo_id,sightline,redshift,grid_path,filters=['lsst_g','lsst_r','lsst_i','lsst_z'],apply_dust=True,plot=True,verbose=True):

        halo_info,stars,centres = self.cluster_stars(halo_id, alpha=0.86, linking_length=20, min_stars=10, mass_frac=0.9, sig_frac=0.2, plot=plot,verbose=verbose)

        stars,tCentres = self.resolve_clusters(sightline=sightline,redshift=redshift,stars=stars,centres=centres,lsst_fwhm=0.7,merge_threshold = 0.3,plot=plot,verbose=verbose)

        clusters,masses = self.extract_clusters(stars,max_radius=30)

        galaxy_app_mags = []
        galaxy_abs_mags = []
        for cluster in clusters:
            proc_cluster = self.cluster_mags(cluster,redshift,grid_path,filters,apply_dust)
            galaxy_app_mags.append(proc_cluster['ApparentMags'])
            galaxy_abs_mags.append(proc_cluster['AbsoluteMags'])

        halo_info['GalaxyStellarMasses'] = masses
        halo_info['GalaxyXY'] = tCentres

        ngals = len(masses)
        halo_info['GalaxyApparentMags'] = {key: np.array([galaxy_app_mags[i][key] for i in range(ngals)]) for key in filters}
        halo_info['GalaxyAbsoluteMags'] = {key: np.array([galaxy_abs_mags[i][key] for i in range(ngals)]) for key in filters}

        M_sun_g = 5.12
        luminosities = 10**((M_sun_g - halo_info['GalaxyAbsoluteMags']['lsst_g']) / 2.5)

        halo_info['GalaxyMassLightRatio'] = masses/luminosities

        del(halo_info['StarsParticleIDs'])
        
        return halo_info