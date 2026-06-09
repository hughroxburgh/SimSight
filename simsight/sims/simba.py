import caesar
from readgadget import readsnap
import yt
yt.funcs.mylog.setLevel(50)
from time import time
from tqdm import tqdm

import numpy as np
from astropy.cosmology import FlatLambdaCDM

class SIMBA_SightlineSim():
    def __init__(self,data_path,fsps_path=None):

        self.data_path = data_path
        sim = self._SimbaCaesarManipulation(data_path)
        self.hub = sim.simulation.hubble_constant
        self.cosmo = FlatLambdaCDM(H0=sim.simulation.hubble_constant*100, Om0=sim.simulation.omega_matter,Ob0=sim.simulation.omega_baryon)
        self.box_size = sim.simulation.boxsize.value+0
        self.name = 'SIMBA'
        self.kernel = 1
        self.f_igm = 0.87 # Khyrkin+2024

        if fsps_path is None:
            self.fsps_path = '/idia/projects/simba/rhoxu/fsps'

        self._halo_cache = None

        self.get_redshifts()

    def _get_snap_num(self,snapNum):

        return 151-snapNum

    def get_redshifts(self):

        redshifts = np.array([0,1.6742000e-02, 3.3612000e-02, 5.0617000e-02, 6.7760000e-02, 8.5046000e-02, 1.0247900e-01, 1.2006500e-01,
                        1.3780700e-01, 1.5571200e-01, 1.7378400e-01, 1.9202900e-01, 2.1045100e-01, 2.2905700e-01, 2.4785200e-01, 2.6684200e-01,
                        2.8603400e-01, 3.0543200e-01, 3.2504400e-01, 3.4487600e-01, 3.6493400e-01, 3.8522500e-01, 4.0575700e-01, 4.2653700e-01,
                        4.4757100e-01, 4.6886800e-01, 4.9043500e-01, 5.1228000e-01, 5.3441200e-01, 5.5683900e-01, 5.7956900e-01, 6.0261200e-01,
                        6.2597600e-01, 6.4967200e-01, 6.7370900e-01, 6.9809600e-01, 7.2284500e-01, 7.4796500e-01, 7.7346900e-01, 7.9936600e-01,
                        8.2566900e-01, 8.5238900e-01, 8.7954000e-01, 9.0713200e-01,  9.3518100e-01, 9.6369900e-01, 9.9270000e-01, 1.0221980e+00,
                        1.0522080e+00, 1.0827470e+00, 1.1138280e+00, 1.1454700e+00, 1.1776880e+00, 1.2105010e+00, 1.2439260e+00, 1.2779810e+00,
                        1.3126870e+00, 1.3480630e+00, 1.3841300e+00, 1.4209090e+00, 1.4584220e+00, 1.4966930e+00, 1.5357450e+00, 1.5756030e+00,
                        1.6162920e+00, 1.6578390e+00, 1.7002710e+00, 1.7436170e+00, 1.7879060e+00, 1.8331690e+00, 1.8794390e+00, 1.9267470e+00,
                        1.9751290e+00, 2.0246210e+00, 2.0752590e+00, 2.1270830e+00, 2.1801320e+00, 2.2344490e+00, 2.2900770e+00, 2.3470620e+00,
                        2.4054520e+00, 2.4652950e+00, 2.5266440e+00, 2.5895520e+00,  2.6540760e+00, 2.7202730e+00, 2.7882060e+00, 2.8579380e+00,
                        2.9295360e+00, 3.0030700e+00, 3.0786140e+00, 3.1562440e+00, 3.2360400e+00, 3.3180860e+00, 3.4024710e+00, 3.4892870e+00,
                        3.5786300e+00, 3.6706020e+00, 3.7653110e+00, 3.8628680e+00,  3.9633920e+00, 4.0670060e+00, 4.1738410e+00, 4.2840340e+00,
                        4.3977300e+00, 4.5150820e+00, 4.6362510e+00, 4.7614050e+00, 4.8907250e+00, 5.0244000e+00, 5.1626300e+00, 5.3056270e+00,
                        5.4536150e+00, 5.6068300e+00, 5.7655260e+00, 5.9299680e+00, 6.1004390e+00, 6.2772410e+00, 6.4606930e+00, 6.6511360e+00,
                        6.8489310e+00, 7.0544630e+00, 7.2681450e+00, 7.4904150e+00,   7.7217420e+00, 7.9626280e+00, 8.2136080e+00, 8.4752570e+00,
                        8.7481910e+00, 9.0330710e+00, 9.3306070e+00, 9.6415620e+00, 9.9667590e+00, 1.0307082e+01, 1.0663489e+01, 1.1037011e+01,
                        1.1428765e+01, 1.1839961e+01, 1.2271910e+01, 1.2726037e+01, 1.3203888e+01, 1.3707152e+01, 1.4237665e+01, 1.4797437e+01,
                        1.5388664e+01, 1.6013753e+01, 1.6675347e+01, 1.7376352e+01, 1.8119968e+01, 1.8909730e+01, 1.9749544e+01, 9.9000000e+01])

        self.redshifts = redshifts
    
    def _SimbaSnapManipulation(self,snapNum):
        """
        Used for going from snapshot file to next z snapshot (ie. 151 to 150).
        """

        parts = self.data_path.split('_')
        if 9 < snapNum < 100:
            snapNum = f'0{snapNum}'
        elif snapNum < 10:
            snapNum = f'00{snapNum}'
        new_snap_file = f'{parts[0]}_{parts[1]}_{snapNum}.hdf5'

        return new_snap_file

    def _SimbaCaesarManipulation(self,snapFile):
        """
        Used for loading Caesar files.
        """

        caesar_sim_file = ''
        for ting in snapFile.split('/')[1:]:
            if ting[:4] == 'snap':
                ting = f'Groups/{ting[5:]}'
            caesar_sim_file += f'/{ting}'
        sim = caesar.load(caesar_sim_file)

        return sim
    
    def load_data(self,particle_type,fields,snapNum,method=None):
        """
        Load data.
        """

        fieldTransfer = {'Coordinates':'pos',
                         'Density':'rho',
                         'ElectronAbundance':'ne',
                         'StarFormationRate':'sfr',
                         'Masses': 'mass',
                         'SmoothingLength':'hsml',
                         'ParticleIDs' : 'pid',
                         'Metallicity' : 'z',
                         'StellarFormationTime' : 'age'}

        snapFile = self._SimbaSnapManipulation(snapNum)

        data = {}
        for field in fields:
            truefield = fieldTransfer[field]
            d = readsnap(snapFile,truefield,particle_type,units=0,suppress=1)
            data[field] = d

        if 'Coordinates' in fields:
            data['Coordinates'] = data['Coordinates'] / self.hub
        if 'Density' in fields:
            data['Density'] = data['Density'] * self.hub**2


        return data
    
    def _load_particle_ids(self,snap_num,stars=True,gas=True):

        print('Loading particle ID cache...', end='\r')
        ts = time()

        if self._halo_cache is None or self._halo_cache['snapshot'] != snap_num:
            self.load_halos(snap_num,return_dict=False)

        if stars:
            self._halo_cache['star_ids'] = self.load_data('stars',['ParticleIDs'],snap_num)['ParticleIDs']
        if gas:
            self._halo_cache['gas_ids'] = self.load_data('gas',['ParticleIDs'],snap_num)['ParticleIDs']

        print(f'Loading particle ID cache -- done! ({time()-ts:.1f}s)')

    def load_halos(self,snap_num,return_dict=True,cache_stars_particles=False, cache_gas_particles=False):

        snapFile = self._SimbaSnapManipulation(snap_num)
        sim = self._SimbaCaesarManipulation(snapFile)

        self._halo_cache = {'snapshot': snap_num,
                            'halos_full': sim.halos}

        if cache_stars_particles or cache_gas_particles:
            self._load_particle_ids(snap_num,stars=cache_stars_particles,gas=cache_stars_particles)

        if return_dict:
            n = len(sim.halos)
            halos = []
            for i in tqdm(range(n),desc=f'Generating snap {snap_num} halos dict'):
                halo = sim.halos[i]
                halos.append({
                    'ID': i,
                    'Pos' : halo.minpotpos.value.astype(np.float32),
                            'Radius' : halo.virial_quantities['r200c'].value.astype(np.float32),
                            'TotalMass' : halo.virial_quantities['m200c'].value.astype(np.float32),
                            'GasMass' : halo.masses['gas'].value.item(),
                            'StellarMass' : halo.masses['stellar'].value.item(),
                            'NumStars' : halo.nstar
                            # 'GalaxyIDs' : halo.galaxy_index_list
                })

            return halos
        
    def load_halo(self,snap_num,halo_id,load_stars=True,load_gas=True): 

        if self._halo_cache is None or self._halo_cache['snapshot'] != snap_num:
            self.load_halos(snap_num,return_dict=False)
            
        c = self._halo_cache
        halo = c['halos_full'][halo_id]

        halo_dict = {'ID' : halo_id,
                     'Pos' : halo.minpotpos.value.astype(np.float32),
                     'Radius' : halo.virial_quantities['r200c'].value.astype(np.float32),
                     'TotalMass' : halo.virial_quantities['m200c'].value.astype(np.float32),
                     'GasMass' : halo.masses['gas'].value.item(),
                     'StellarMass' : halo.masses['stellar'].value.item(),
                     "NumStars" : halo.nstar
                    #  'GalaxyIDs' : halo.galaxy_index_list
                     }
        
        if load_stars:
            halo_dict['StarsParticleIDs'] = c['star_ids'][halo.slist]

        if load_gas:
            halo_dict['GasParticleIDs'] = c['gas_ids'][halo.glist]
        
        return halo_dict
    
    # def load_galaxy(self,snap_num,galaxy_id): #,particles_only=False):

    #     if snap_num != self._halos_full_snap:
    #         self.load_halos(snap_num,return_dict=False)

    #     galaxy = self.galaxies[galaxy_id]

    #     # if particles_only:
    #     #     return galaxy.glist

    #     galaxy_dict = {'ID' : galaxy_id,
    #                  'Pos' : galaxy.pos.value.astype(np.float32),
    #                  'Radius' : np.float32(galaxy.radii['baryon_r80'].value),
    #                  'Mass' : galaxy.mass.value.item(),
    #                  'ParticleIDs' : self._snap_particle_ids[galaxy.glist],
    #                  'ParentHaloID' : galaxy.parent_halo_index}
        
    #     return galaxy_dict
    
    def radius_mapping(self,data):
        """
        Returns effective radii for finding points of impact to ray.
        """

        return data['SmoothingLength'] * 2

    
    def process_data(self,transformed_points,data,length):

        from ._sph_compute import Find_Intersection_Intervals, Build_Sparse_Weights, Calculate_Field, Nearest_Particle

        t_res = 10 #ckpc

        # midpoints of full cells
        t_grid = np.arange(t_res/2, length, t_res)
        lengths = np.full_like(t_grid, t_res, dtype=float)

        # append last partial segment if it exists
        if length % t_res > 0:
            t_grid = np.append(t_grid, length)          # sample at exact endpoint
            lengths = np.append(lengths, length % t_res)     # segment length = leftover

        particle_intervals = Find_Intersection_Intervals(transformed_points,data['SmoothingLength'])

        weight_matrix = Build_Sparse_Weights(t_grid,transformed_points,data['SmoothingLength'],
                                             particle_intervals[:,0].astype(int),particle_intervals[:,1],particle_intervals[:,2],self.kernel)

        new_data = {}
        volumes = data['Masses']/data['Density']

        for field in data.keys():
            if field in ['Masses','SmoothingLength']:
                continue

            new_data[field] = np.zeros_like(t_grid)

            scalar_field = data[field] if field != 'Density' else None

            try:
                Calculate_Field(new_data[field],weight_matrix[0],weight_matrix[1],weight_matrix[2],data['Masses'],volumes,
                                scalar_field=scalar_field, is_sph=False)
            
            except:
                e = f'Failed here: weight[0] = {weight_matrix[0]}\n\ weight[1] = {weight_matrix[1]}\n\ weight[2] = {weight_matrix[2]}\n SL length = {length}'
                raise ValueError(e)
            
        ids = Nearest_Particle(t_grid,transformed_points)
            
        return new_data, lengths, ids