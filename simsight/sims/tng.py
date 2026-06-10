import illustris_python as il
import numpy as np
from time import time as clock
from astropy.cosmology import FlatLambdaCDM
from tqdm import tqdm

from ._arepo_compute import Find_Line_Elements
from .._utils import _Progress_Print

class TNG_SightlineSim():
    def __init__(self,data_path,fsps_path=None):

        self.data_path = data_path
        header = il.groupcat.loadHeader(data_path,99)
        self.hub = header['HubbleParam']
        self.cosmo = FlatLambdaCDM(H0=self.hub*100, Om0=header['Omega0'],Ob0=0.0486)
        self.box_size = header['BoxSize']/self.hub
        self.name = 'TNG'
        self.f_igm = 0.8195   #0.85 Martizzi+2019, but not super accurate

        if fsps_path is None:
            self.fsps_path = 'fsps'

        self._halo_cache = None

        self._get_redshifts()

    def _get_snap_num(self,snapNum):

        return 99-snapNum
    
    def _get_redshifts(self):

        # redshifts = []
        # for i in range(99,-1,-1):
        #     header = il.groupcat.loadHeader(self.data_path,i)
        #     redshifts.append(header['Redshift'])
        # redshifts = np.array(redshifts)
        # redshifts[0]=0

        # self.redshifts = redshifts

        self.redshifts = np.array([0, 9.52166697e-03, 2.39744284e-02, 3.37243719e-02,
                                4.85236300e-02, 5.85073228e-02, 7.36613847e-02, 8.38844308e-02,
                                9.94018026e-02, 1.09869940e-01, 1.25759332e-01, 1.41876204e-01,
                                1.52748769e-01, 1.69252033e-01, 1.80385262e-01, 1.97284182e-01,
                                2.14425036e-01, 2.25988386e-01, 2.43540182e-01, 2.61343256e-01,
                                2.73353347e-01, 2.97717685e-01, 3.10074120e-01, 3.28829724e-01,
                                3.47853842e-01, 3.60687657e-01, 3.80167867e-01, 3.99926965e-01,
                                4.19968942e-01, 4.40297849e-01, 4.60917794e-01, 4.81832943e-01,
                                5.03047523e-01, 5.24565820e-01, 5.46392183e-01, 5.75980845e-01,
                                5.98543288e-01, 6.21428745e-01, 6.44641841e-01, 6.76110411e-01,
                                7.00106354e-01, 7.32636182e-01, 7.57441373e-01, 7.91068249e-01,
                                8.16709979e-01, 8.51470901e-01, 8.86896938e-01, 9.23000816e-01,
                                9.50531352e-01, 9.97294226e-01, 1.03551045e+00, 1.07445789e+00,
                                1.11415056e+00, 1.15460271e+00, 1.20625808e+00, 1.24847261e+00,
                                1.30237846e+00, 1.35757667e+00, 1.41409822e+00, 1.49551217e+00,
                                1.53123903e+00, 1.60423452e+00, 1.66666956e+00, 1.74357057e+00,
                                1.82268925e+00, 1.90408954e+00, 2.00202814e+00, 2.10326965e+00,
                                2.20792547e+00, 2.31611074e+00, 2.44422570e+00, 2.57729027e+00,
                                2.73314262e+00, 2.89578501e+00, 3.00813107e+00, 3.28303306e+00,
                                3.49086137e+00, 3.70877426e+00, 4.00794511e+00, 4.17683491e+00,
                                4.42803374e+00, 4.66451770e+00, 4.99593347e+00, 5.22758097e+00,
                                5.52976581e+00, 5.84661375e+00, 6.01075740e+00, 6.49159775e+00,
                                7.00541705e+00, 7.23627607e+00, 7.59510715e+00, 8.01217295e+00,
                                8.44947629e+00, 9.00233985e+00, 9.38877127e+00, 9.99659047e+00,
                                1.09756433e+01, 1.19802133e+01, 1.49891732e+01, 2.00464910e+01])
    

    def _load_chunks_efficient(self, snap_num, part_type, true_fields):
        """
        Two-pass HDF5 loader. Pre-allocates output arrays then fills
        directly, avoiding the concatenation memory spike.
        """
        import h5py

        part_type_map = {'gas':        0,
                 'dm':         1,
                 'tracers':    3,
                 'stars':      4,
                 'blackholes': 5}

        part_type = part_type_map.get(part_type, part_type)

        pkey = f"PartType{part_type}"

        # -- Get number of chunk files from first header -- #
        dtypes  = {}
        ndims   = {}
        with h5py.File(il.snapshot.snapPath(self.data_path, snap_num, 0), 'r') as f:
            n_chunks = int(f['Header'].attrs['NumFilesPerSnapshot'])
            for field in true_fields:
                if field in f.get(pkey, {}):
                    ds            = f[pkey][field]
                    dtypes[field] = ds.dtype
                    ndims[field]  = ds.shape[1] if ds.ndim > 1 else None

        # -- Pass 1: accumulate total N and read dtypes/shapes -- #
        total_n = 0
        for chunk in range(n_chunks):
            with h5py.File(il.snapshot.snapPath(self.data_path, snap_num, chunk), 'r') as f:
                total_n += int(f['Header'].attrs['NumPart_ThisFile'][part_type])

        # -- Pre-allocate final arrays once -- #
        data = {}
        for field in true_fields:
            if field not in dtypes:
                continue
            dtype        = np.float32 if dtypes[field] == np.float64 else dtypes[field]
            shape        = (total_n, ndims[field]) if ndims[field] else (total_n,)
            data[field]  = np.empty(shape, dtype=dtype)

        # -- Pass 2: fill directly into pre-allocated arrays -- #
        offset = 0
        for chunk in tqdm(range(n_chunks),desc='Loading data'):
            with h5py.File(il.snapshot.snapPath(self.data_path, snap_num, chunk), 'r') as f:
                if pkey not in f:
                    continue
                n = int(f['Header'].attrs['NumPart_ThisFile'][part_type])
                if n == 0:
                    continue
                for field in true_fields:
                    if field in data and field in f[pkey]:
                        data[field][offset:offset + n] = f[pkey][field][:]
                offset += n

        return data


    def load_data(self, particle_type, fields, snapNum,method='custom'):

        fieldTransfer = {'Coordinates':          'Coordinates',
                        'Density':              'Density',
                        'ElectronAbundance':    'ElectronAbundance',
                        'StarFormationRate':    'StarFormationRate',
                        'Masses':               'Masses',
                        'SmoothingLength':      'SmoothingLength',
                        'ParticleIDs':          'ParticleIDs',
                        'Metallicity':          'GFM_Metallicity',
                        'StellarInitialMass':   'GFM_InitialMass',
                        'StellarFormationTime': 'GFM_StellarFormationTime'}

        # SmoothingLength is not in HDF5 for moving-mesh sims — skip it
        truefields = [fieldTransfer[f] for f in fields
                    if fieldTransfer[f] != 'SmoothingLength']

        if method == 'custom':
            raw  = self._load_chunks_efficient(snapNum, particle_type, truefields)
        elif method == 'illustris':
            raw = il.snapshot.loadSubset(self.data_path,snapNum,particle_type,truefields)
            if len(truefields) == 1:
                raw = {truefields[0]:raw}

        data = {f: raw[fieldTransfer[f]] for f in fields if fieldTransfer[f] in raw}

        # -- Unit conversions -- #
        if 'Coordinates' in data:
            data['Coordinates']     /= self.hub
        if 'Density' in data:
            data['Density']         *= self.hub**2
        if 'Masses' in data:
            data['Masses']          /= self.hub
        if 'StellarInitialMass' in data:
            data['StellarInitialMass'] /= self.hub

        return data
    






    # def load_halos(self,snap_num):

    #     halos_full = il.groupcat.loadHalos(self.data_path,snap_num,fields=['GroupPos','Group_R_Crit200','GroupMassType','GroupFirstSub','GroupNsubs'])
    #     halos = []
    #     for i in range(halos_full['count']):
    #         halo_dict = {'ID' : i,
    #                     'Pos' : halos_full['GroupPos'][i] / self.hub,
    #                     'Radius' : halos_full['Group_R_Crit200'][i] / self.hub,
    #                     'TotalMass' : np.nansum(halos_full['GroupMassType'][i]) * 1e10 / self.hub,
    #                     'GasMass' : halos_full['GroupMassType'][i,0] * 1e10 / self.hub,
    #                     'StellarMass' : halos_full['GroupMassType'][i,4] * 1e10 / self.hub,    # In Msun
    #                     'GalaxyIDs' : np.array([halos_full['GroupFirstSub'][i] + j for j in range(halos_full['GroupNsubs'][i])])}       
    #         halos.append(halo_dict)

    #     halos = np.array(halos)
    #     return halos

    def _load_particle_ids(self,snap_num,stars=True,gas=True):

        msg = 'Loading particle ID cache'
        print(msg, end='\r')
        ts = clock()

        if self._halo_cache is None or self._halo_cache['snapshot'] != snap_num:
            self.load_halos(snap_num,return_dict=False)

        if stars:
            all_star_ids = il.snapshot.loadSubset(self.data_path, snap_num, partType=4, fields=['ParticleIDs'])
            star_offsets = np.concatenate([[0], np.cumsum(self._halo_cache['halos_full']['GroupLenType'][:, 4])])
            self._halo_cache['star_ids'] = all_star_ids
            self._halo_cache['star_offsets'] = star_offsets
        if gas:
            all_gas_ids = il.snapshot.loadSubset(self.data_path, snap_num, partType=0, fields=['ParticleIDs'])
            gas_offsets = np.concatenate([[0], np.cumsum(self._halo_cache['halos_full']['GroupLenType'][:, 0])])
            self._halo_cache['gas_ids'] = all_gas_ids
            self._halo_cache['gas_offsets'] = gas_offsets

        _Progress_Print(msg,ts)

    def load_halos(self, snap_num, return_dict=True, cache_stars_particles=False, cache_gas_particles=False):

        halos_full = il.groupcat.loadHalos(self.data_path, snap_num, fields=[
            'GroupPos', 'Group_R_Crit200', 'Group_M_Crit200', 'GroupMassType', 'GroupFirstSub', 'GroupNsubs', 'GroupLenType'
        ])

        self._halo_cache = {'snapshot': snap_num,
                            'halos_full': halos_full}


        if cache_stars_particles or cache_gas_particles:
            self._load_particle_ids(snap_num,stars=cache_stars_particles,gas=cache_stars_particles)

        if return_dict:
            n = halos_full['count']
            halos = []
            for i in tqdm(range(n),desc=f'Generating snap {snap_num} halos dict'):
                halos.append({
                    'ID': i,
                    'Pos': halos_full['GroupPos'][i].astype(np.float32) / self.hub,
                    'Radius': halos_full['Group_R_Crit200'][i].astype(np.float32) / self.hub,
                    'TotalMass': halos_full['Group_M_Crit200'][i].astype(np.float32) * 1e10 / self.hub,
                    'GasMass': halos_full['GroupMassType'][i, 0].astype(np.float32) * 1e10 / self.hub,
                    'StellarMass': halos_full['GroupMassType'][i, 4].astype(np.float32) * 1e10 / self.hub,
                    'NumStars': halos_full['GroupLenType'][i, 4]
                    # 'GalaxyIDs': np.arange(halos_full['GroupFirstSub'][i],
                    #                     halos_full['GroupFirstSub'][i] + halos_full['GroupNsubs'][i])
                })

            return np.array(halos)
    
    # def load_halo(self,snap_num,halo_id): #,particles_only=False):

    #     gas_halo_pts = il.snapshot.loadHalo(self.data_path,snap_num,id=halo_id,partType='gas',fields=['ParticleIDs'])
    #     stars_halo_pts = il.snapshot.loadHalo(self.data_path,snap_num,id=halo_id,partType='stars',fields=['ParticleIDs'])
    #     # if particles_only:
    #     #     return halo_pts

    #     halo_info = il.groupcat.loadSingle(self.data_path,snap_num,haloID=halo_id)
        
    #     halo_dict = {'ID' : halo_id,
    #                  'Pos' : halo_info['GroupPos'] / self.hub,
    #                  'Radius' : halo_info['Group_R_Crit200'] / self.hub,
    #                  'TotalMass' : np.nansum(halo_info['GroupMassType']) * 1e10 / self.hub,
    #                  'GasMass' : halo_info['GroupMassType'][0] * 1e10 / self.hub,
    #                  'StellarMass' : halo_info['GroupMassType'][4] * 1e10 / self.hub,    # In Msun
    #                  'GasParticleIDs' : gas_halo_pts,
    #                  'StarsParticleIDs' : stars_halo_pts,
    #                  'GalaxyIDs' : np.array([halo_info['GroupFirstSub'] + j for j in range(halo_info['GroupNsubs'])])}

    #     return halo_dict

    def load_halo(self, snap_num, halo_id,load_stars=True,load_gas=True):

        if self._halo_cache is None or self._halo_cache['snapshot'] != snap_num:
            self.load_halos(snap_num,return_dict=False)
        
        c = self._halo_cache
        hf = c['halos_full']

        halo_dict = {
            'ID': halo_id,
            'Pos': hf['GroupPos'][halo_id].astype(np.float32) / self.hub,
            'Radius': hf['Group_R_Crit200'][halo_id].astype(np.float32) / self.hub,
            'TotalMass': np.nansum(hf['GroupMassType'][halo_id]).astype(np.float32) * 1e10 / self.hub,
            'GasMass': hf['GroupMassType'][halo_id, 0].astype(np.float32) * 1e10 / self.hub,
            'StellarMass': hf['GroupMassType'][halo_id, 4].astype(np.float32) * 1e10 / self.hub,
            'NumStars': hf['GroupLenType'][halo_id, 4]
            # 'GalaxyIDs': np.arange(hf['GroupFirstSub'][halo_id],
            #                     hf['GroupFirstSub'][halo_id] + hf['GroupNsubs'][halo_id])
        }

        if load_gas:
            if 'gas_ids' in c:
                gas_start = c['gas_offsets'][halo_id]
                gas_end = c['gas_offsets'][halo_id + 1]
                halo_dict['GasParticleIDs'] = c['gas_ids'][gas_start:gas_end]   
            else:
                halo_dict['GasParticleIDs'] = il.snapshot.loadHalo(self.data_path, snap_num, id=halo_id, partType='gas', fields=['ParticleIDs'])

        if load_stars:
            if 'star_ids' in c:
                star_start = c['star_offsets'][halo_id]
                star_end = c['star_offsets'][halo_id + 1]
                halo_dict['StarsParticleIDs'] = c['star_ids'][star_start:star_end]
            else:
                halo_dict['StarsParticleIDs'] = il.snapshot.loadHalo(self.data_path, snap_num, id=halo_id, partType='stars', fields=['ParticleIDs'])

        return halo_dict
    
    # def load_galaxy(self,snap_num,galaxy_id): #,particles_only=False):

    #     galaxy_pts = il.snapshot.loadSubhalo(self.data_path,snap_num,id=galaxy_id,partType='gas',fields=['ParticleIDs'])
    #     # if particles_only:
    #     #     return galaxy_pts

    #     galaxy_info = il.groupcat.loadSingle(self.data_path,snap_num,subhaloID=galaxy_id)

    #     galaxy_dict = {'ID' : galaxy_id,
    #                  'Pos' : galaxy_info['SubhaloPos'] / self.hub,
    #                  'Radius' : galaxy_info['SubhaloHalfmassRad'] / self.hub,
    #                  'Mass' : galaxy_info['SubhaloMass'] * 1e10 / self.hub,
    #                  'ParticleIDs' : galaxy_pts,
    #                  'ParentHaloID' : galaxy_info['SubhaloGrNr']}
    #     return galaxy_dict
    
    def radius_mapping(self,data):
        """
        Returns effective radii for finding points of impact to ray.
        """

        return 3*(3/(4*np.pi)*data['Masses']/data['Density'])**(1/3)

    
    def process_data(self,transformed_points,data,length):

        cell_ids,cell_line_elements = Find_Line_Elements(transformed_points,length)

        new_data = {}

        for field in data.keys():
            if field in ['Masses']:
                continue

            new_data[field] = data[field][cell_ids]
            
        return new_data, cell_line_elements, cell_ids
