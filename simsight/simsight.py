import multiprocessing
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

import warnings
warnings.filterwarnings('ignore')
import sys
from glob import glob
import os
from tqdm import tqdm

from time import time as clock
import numpy as np

from .sims import load_sim
from ._visualiser_class import VisualSim
from ._utils import _Progress_Print, _Smart_Tqdm, _Is_Interactive,Cleanup_Memory

class SightlineSim():

    def __init__(self,data_path,num_cores=None,backend='threading',
                 snap_path_structure=None,halo_path_structure=None,fsps_path=None):
        """
        data_path : where the data is stored on the servers. As long as the words 'TNG' or 'SIMBA' are inside, the class will try to conform to those standards.
        parallel : run things in parallel -- SEEMS TO BE WORSE SO JUST LEAVE FALSE FOR NOW
        """

        # -- Set up init -- #
        self.data_path = data_path  
        self.backend = backend
        if num_cores is None:
            self.num_cores = int(multiprocessing.cpu_count())
        else:
            self.num_cores = num_cores

        # -- Select Simulation Suite -- #
        self.sim = load_sim(data_path,snap_path_structure,halo_path_structure,fsps_path)
        print(f'Initialised {self.sim.name}',flush=True)
        
        # -- Visualiser -- #
        self.Vis = VisualSim(self)

        print('Made It')

    
    # ------------- generating / partitioning sightlines ------------- #

    def generate_sightlines(self,redshift,n_sightlines=1,method='random',origin=None,direction_vector=None):
        """
        Generates desired number of Sightline objects with either random or healpix direction distribution.
        """

        from ._sightline_class import Sightline

        if n_sightlines == 1:
            if origin is None:
                origin = np.random.rand(3).astype(np.float32)*self.sim.box_size 
            if direction_vector is None:
                direction_vector = np.random.rand(3).astype(np.float32) * 2 - 1
                direction_vector /= np.linalg.norm(direction_vector)
            
            return Sightline(origin=origin,direction_vector=direction_vector,target_redshift=redshift)

        else:
            if method == 'random':
                origins = np.random.rand(n_sightlines,3).astype(np.float32)*self.sim.box_size    
                directions = np.random.rand(n_sightlines,3).astype(np.float32) * 2 - 1
                directions /= np.linalg.norm(directions)

            elif method == 'fullsky':
                
                import healpy as hp

                nside = np.sqrt(n_sightlines/12)
                nside = np.round(nside).astype(int)
                n = hp.nside2npix(nside)
                ipix = np.arange(n)
                x, y, z = hp.pix2vec(nside, ipix)
                directions = np.column_stack([x, y, z]).astype(np.float32)
                n_sightlines = len(directions)
                if origin is None:
                    origin = np.random.rand(1,3).astype(np.float32)*self.sim.box_size
                origins = np.tile(origin, (n_sightlines, 1)).astype(np.float32)
                print(f'N HEALpix sightlines = {n_sightlines}, origin = {origin}',flush=True)

            sightlines = []
            for i in range(n_sightlines):
                SL = Sightline(origin=origins[i],direction_vector=directions[i],target_redshift=redshift)
                SL.sightline_idx = i
                sightlines.append(SL)

            return sightlines
    

    def _generate_and_partition_sightlines(self,n_sightlines,redshift,parallel=True,method='random',origin=None):
        """
        Generate sightlines and partition them.
        """

        # -- Generate sightlines -- #
        sightlines = self.generate_sightlines(redshift,n_sightlines,method,origin)

        # -- Partition first sightline with verbose=True to display how many snapshots needed -- #
        sightlines[0].partition(self.sim,verbose=True)

        # -- Partition remaining sightlines -- #
        if parallel:
            sightlines[1:] = Parallel(n_jobs=self.num_cores,backend=self.backend)(
                delayed(sl.partition)(self.sim,verbose=False) for sl in _Smart_Tqdm(sightlines[1:],desc='Creating all sightlines')
                )
        else:

            for sl in _Smart_Tqdm(sightlines[1:],desc='Creating all sightlines'):
                sl.partition(self.sim,verbose=False)

        return sightlines
    

    # ------------- Loading / saving sightlines ------------- #

    def load_sightlines(self,directory_path=None,percent=100,sl_files=None,with_pidx=False):

        import pickle
        from glob import glob

        if directory_path is not None and sl_files is not None:
            raise ValueError('"directory_path" and "sl_path" cannot both be provided!')
        
        elif directory_path is not None:
            if os.path.exists(directory_path):

                if not _Is_Interactive():
                    msg = f"Loading sightlines"
                    ts = clock()
                    print(msg,end='\r',flush=True)

                files = sorted(glob(f'{directory_path}/*.pkl'))
                if len(files) > 0:
                    n_files = int(percent*len(files)/100)
                    sightlines = []
                    for file in _Smart_Tqdm(files[:n_files],desc='Loading sightlines'):
                        sightline_idx = int(file.split('_')[-1].split('.pkl')[0])
                        with open(file,'rb') as f:
                            SL = pickle.load(f)
                            SL.sightline_idx = sightline_idx

                        if with_pidx:
                            if all(SL.sub_PointsIdx[i] == ['Removed'] for i in range(SL.subsightline_reached(grid=True,halos=True))):
                                path = f'{directory_path}/sightline_{sightline_idx}_PointsIdx.npy'
                                if os.path.exists(path):
                                    SL.sub_PointsIdx = np.load(path,allow_pickle=True).tolist()
                                else:
                                    e = f'No stored PointsIdx found for sightline {sightline_idx}'
                                    raise FileExistsError(e)

                        sightlines.append(SL)


                    
                    if not _Is_Interactive():
                        _Progress_Print(msg,ts)
                        
                    return np.array(sightlines)
            else:
                print('No sightlines saved in directory_path.',flush=True)
                return None
            
        else:
            n_files = int(percent*len(sl_files)/100)
            sightlines = []
            for file in _Smart_Tqdm(sl_files[:n_files],desc='Loading sightlines'):
                sightline_idx = int(file.split('_')[-1].split('.pkl')[0])
                with open(file,'rb') as f:
                    SL = pickle.load(f)
                    SL.sightline_idx = sightline_idx
                    sightlines.append(SL)

            return np.array(sightlines)
                    

    def save_sightlines(self, sightlines, save_path):

        if not _Is_Interactive():
            msg = f"    saving sightlines to {save_path}"
            ts = clock()
            print(msg, end='\r', flush=True)

        files = glob(f'{save_path}/*.pkl')

        def _save_one(sl):
            return sl.save(save_path, return_file=True)

        saved_files = Parallel(n_jobs=self.num_cores, backend='loky')(
            delayed(_save_one)(sl)
            for sl in _Smart_Tqdm(sightlines, desc='    saving sightlines')
        )

        get_reusable_executor().shutdown(wait=True)

        saved_set = set(saved_files)
        for file in files:
            if file not in saved_set:
                os.remove(file)

        if not _Is_Interactive():
            _Progress_Print(msg, ts)

        Cleanup_Memory()


    # ------------- Computation architecture ------------- #

    def _choose_function(self,f):

        from ._compute import Calc_Ray_Density, Calc_Ray_DM

        mapping = {'Density': {'func': Calc_Ray_Density, 
                               'fields': ['Coordinates','Density','Masses','SmoothingLength']},

                    'DM' : {'func': Calc_Ray_DM, 
                            'fields': ['Coordinates','Density','Masses','SmoothingLength','StarFormationRate','ElectronAbundance']}}
        
        func = mapping[f]['func']
        fields = mapping[f]['fields']
        fields = [f for f in fields if f not in self.sim.point_find_fields]

        return func,fields

    def _finder_architecture(self,data,findtype,percentile=99.9):

        # -- Define particle radii and the maximum radius to search within -- #
        radii = self.sim.radius_mapping(data)
        coarse_radius = np.percentile(radii,percentile)
        giant_bool = radii > coarse_radius
        giant_idx = np.where(giant_bool)[0]
        giant_pts = data['Coordinates'][giant_idx]
        giant_radii = radii[giant_idx]

        ts = clock()
        if findtype == 'tree':
                
            from scipy.spatial import cKDTree

            #  -- Generate KDTree of all points in simulation -- #
            msg = f"    generating KDTree"
            print(msg,end='\r')
            tree = cKDTree(data['Coordinates'])
            _Progress_Print(msg,ts)

            return tree, radii, coarse_radius, giant_idx, giant_pts, giant_radii

        elif findtype == 'voxel':

            from ._utils import _Counting_Sort

            msg = f"    generating voxelgrid"
            print(msg,end='\r')
            voxel_size = coarse_radius
            grid_size = int(np.ceil(self.sim.box_size / voxel_size))

            # -- Compute flat keys column by column — avoids storing full (N,3) ijk array -- #
            flat  = (data['Coordinates'][:, 0] / voxel_size).astype(np.int32)
            flat *= grid_size * grid_size                              # i * grid_size²
            tmp   = (data['Coordinates'][:, 1] / voxel_size).astype(np.int32)
            flat += tmp * grid_size; del tmp                           # + j * grid_size
            tmp   = (data['Coordinates'][:, 2] / voxel_size).astype(np.int32)
            flat += tmp;             del tmp  

            # -- argsort: order[i] is directly the global particle index -- #
            flat[giant_bool] = -1
            order, offsets = _Counting_Sort(flat, grid_size**3)
            sorted_flat = flat[order]
            del flat

            first_normal = int(np.searchsorted(sorted_flat, 0))

            boundaries   = np.concatenate([[first_normal],
                                np.where(np.diff(sorted_flat[first_normal:]))[0] + 1 + first_normal,
                                [len(sorted_flat)]])
            
            unique_flat  = sorted_flat[boundaries[:-1]]
            del sorted_flat

            voxels = {int(k): order[s:e].copy()
                        for k, s, e in zip(unique_flat, boundaries[:-1], boundaries[1:])
                        if k >= 0}
            del order

            _Progress_Print(msg,ts)

            Cleanup_Memory()

            return {'voxels':voxels,
                    'coords':data['Coordinates'],
                    'grid_size':grid_size}, radii, coarse_radius, giant_idx, giant_pts, giant_radii
        


    # ------------- Find points / halos in given sightlines ------------- #

    def _snapshot_points_in_sightlines(self,sightlines,snapshot,architecture,radii,coarse_radius,findtype,
                                       giant_idx,giant_pts,giant_radii,parallel=False):
        
        if not _Is_Interactive():
            msg = f"    finding points in sightlines"
            ts = clock()
            print(msg,end='\r',flush=True)

        from ._point_find import Points_In_Sightline
        
        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(Points_In_Sightline)(sl, snapshot, architecture,radii,coarse_radius,findtype,giant_idx,giant_pts,giant_radii)
                for sl in _Smart_Tqdm(sightlines, desc=f"    finding points in sightlines [snap {snapshot}]")
            )
        else:
            for sl in _Smart_Tqdm(sightlines, desc=f"    finding points in sightlines [snap {snapshot}]"):
                Points_In_Sightline(sl,snapshot,architecture,radii,coarse_radius,findtype,giant_idx,giant_pts,giant_radii)

        if not _Is_Interactive():
            _Progress_Print(msg,ts)


    def _snapshot_compute_sightlines(self, sightlines, data, func, snapshot,parallel=False):
        """
        Run compute in all sightlines for this snapshot.
        """

        from ._compute import Compute_Sightline

        if not _Is_Interactive():
            msg = f"    computing snapshot sightlines"
            ts = clock()
            print(msg,end='\r',flush=True)
        
        if parallel:
            results = Parallel(n_jobs=self.num_cores, backend='threading')(
                delayed(Compute_Sightline)(sl, self.sim, data, func, snapshot)
                for sl in _Smart_Tqdm(sightlines, desc=f"    computing snapshot sightlines [snap {snapshot}]")
            )
        else:
            results = [Compute_Sightline(sl, self.sim, data, func, snapshot) 
                       for sl in _Smart_Tqdm(sightlines, desc=f"    computing snapshot sightlines [snap {snapshot}]")]

        for sl, sl_results in zip(sightlines, results):
            for sub_idx, compute, density, lengths, ids in sl_results:
                sl.sub_Compute[sub_idx] = compute
                sl.sub_Density[sub_idx] = density
                sl.sub_Grid[sub_idx] = lengths
                sl.sub_Cells[sub_idx] = ids

        if not _Is_Interactive():
            _Progress_Print(msg,ts)


    def _snapshot_reduce_sightlines(self, sightlines, save_path,parallel=False):
        """
        Run compute in all sightlines for this snapshot.
        """

        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        if not _Is_Interactive():
            msg = f"    reducing sightlines"
            ts = clock()
            print(msg,end='\r',flush=True)

        def _reduce_one(sl, grid_resolution, cgm_buffer, save_path):

            sl.reduce(grid_resolution=grid_resolution, cgm_buffer=cgm_buffer,
                    save_points_path=save_path)
            
            return sl

        if parallel:
            results = Parallel(n_jobs=self.num_cores, backend='loky',max_nbytes=None)(
                delayed(_reduce_one)(sl, 100, 20, save_path)
                for sl in _Smart_Tqdm(sightlines, desc='    reducing sightlines')
            )

            get_reusable_executor().shutdown(wait=True)

            sightlines[:] = results

        else:
            for i in _Smart_Tqdm(range(len(sightlines)), desc='    reducing sightlines'):
                sightlines[i].reduce(grid_resolution=100,cgm_buffer=20,save_points_path=save_path)
                
        if not _Is_Interactive():
            _Progress_Print(msg,ts)

        Cleanup_Memory()


    def _snapshot_find_halos(self, sightlines, snap, halos, com, radii, tree, max_radius,parallel=False):
        """
        Run compute in all sightlines for this snapshot.
        """

        from ._point_find import Halos_In_Sightline,_Halos_Near_Ray

        # -- Warm up numba kernel (pays JIT cost once, on main thread) -- #
        _Halos_Near_Ray(
            np.zeros(3,          dtype=np.float64),
            np.array([1.,0.,0.], dtype=np.float64),
            1.0,
            com[:1],
            radii[:1],
        )

        if not _Is_Interactive():
            msg = f"    finding halos in sightlines"
            ts = clock()
            print(msg, end='\r', flush=True)

        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(Halos_In_Sightline)(sl, snap, halos, com, radii, tree, max_radius,self.sim.cosmo)
                for sl in _Smart_Tqdm(sightlines, desc='    finding halos in sightlines')
            )
        else:
            for sl in _Smart_Tqdm(sightlines, desc='    finding halos in sightlines'):
                Halos_In_Sightline(sl, snap, halos, com, radii, tree, max_radius,self.sim.cosmo)
            # inactive sightlines untouched — already in sightlines list

        if not _Is_Interactive():
            _Progress_Print(msg, ts)

        Cleanup_Memory()


    # ------------- Main functions ------------- #

    def find_halos_in_sightlines(self, sightlines, parallel=False, num_snaps=None, single_snap=None, announce=True):

        from scipy.spatial import cKDTree

        if single_snap is not None:
            start_snap = single_snap
            snaps_required = single_snap + 1
        else:
            snaps_required = min(v for v in [sl.sub_Snapshots[-1] + 1 for sl in sightlines] + [num_snaps] if v is not None)
            start_snap = min([sl.sub_Snapshots[sl.subsightline_reached(grid=False,halos=True)] for sl in sightlines])

        for snap in range(start_snap, snaps_required):

            halofind_check = snap < min([sl.sub_Snapshots[sl.subsightline_reached(grid=False,halos=True)-1] for sl in sightlines])
            if not halofind_check:
                if single_snap is not None:
                    return False
                else:
                    continue

            trueSnapNum = self.sim._get_snap_num(snap)

            if announce:
                print(f'------Snapshot {snap}------', flush=True)

            # -- Load halos -- #
            halos = self.sim.load_halos(trueSnapNum)

            radii = np.array([h['Radius'] for h in halos], dtype=np.float64)
            com   = np.array([h['Pos']    for h in halos], dtype=np.float64)
            tree       = cKDTree(com)
            max_radius = float(radii.max())

            # -- Find halos in sightlines -- #
            self._snapshot_find_halos(sightlines, snap, halos, com, radii, tree, max_radius,parallel)

            if announce:
                print('\n', flush=True)

            del(halos,radii,com,tree)

        return True

            
    def run_single_sightline(self,sightline=None,redshift=None,origin=None,direction_vector=None,functype='DM',
                             delete_data=True,save_path=None,plot_sightline=False):
        
        from ._compute import Compute_Sightline
        from ._point_find import Points_In_Sightline
        
        # -- Select function to calculate and corresponding data fields needed -- #
        func,fields = self._choose_function(functype)
        
        if save_path is not None:
            sightline = self.load_sightlines(save_path)[0]

        if sightline is None:    
            sightline = self.generate_sightlines(redshift,n_sightlines=1,origin=origin,direction_vector=direction_vector)
            sightline.partition(self.sim)

        snaps_required = len(np.unique(sightline.sub_BoxRedshifts))
        start_snap = sightline.sub_Snapshots[sightline.subsightline_reached(grid=True)]

        for snap in range(start_snap,snaps_required):

            if sightline.subsightline_reached(grid=True) == sightline.num_sub_sightlines or \
                snap < sightline.sub_Snapshots[sightline.subsightline_reached(grid=True)-1]:
                continue

            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)
        
            trueSnapNum = self.sim._get_snap_num(snap)

            point_find_data = self.sim.load_data(particle_type='gas',fields=self.sim.point_find_fields,snapNum=trueSnapNum,method='custom')

            # # -- Load data -- #
            # data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum)

            # -- Define particle radii and the maximum radius to search within -- #
            radii = self.sim.radius_mapping(point_find_data)
            coarse_radius = np.percentile(radii,99.9)

            Points_In_Sightline(sightline,snap,point_find_data['Coordinates'],radii,coarse_radius,findtype='cylinder')

            if (snap == 0) & (plot_sightline):
                self.Vis.plot3d()
                self.Vis.plot_sightline(sightline,data['Coordinates'])

            data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum,method='custom')
            data = data | point_find_data

            results = Compute_Sightline(sightline, self.sim, data, func, snap)

            for sub_idx, compute, density, lengths, ids in results:
                sightline.sub_Compute[sub_idx] = compute
                sightline.sub_Density[sub_idx] = density
                sightline.sub_Grid[sub_idx] = lengths
                sightline.sub_Cells[sub_idx] = ids

            if save_path is not None:
                self.save_sightlines([sightline],save_path)

            if delete_data:
                del(data)


        if delete_data:
            return sightline
        else:
            return sightline, data 

    

    def run_many_sightlines(self,n_sightlines=None,redshift=None,sightlines=None,method='random',origin=None,
                            functype='DM',findtype='tree',load_method='custom',
                            delete_data=True,save_path=None,plot_sightlines=False,reduce_sightlines=False,find_halos=False,save_pointsidx=True,
                            parallel_slgen=False,parallel_findpts=False,parallel_compute=False,parallel_halos=False,parallel_reduce=False):
                            
        """
        Run full loop over chosen number of sightlines.
        """

        if sightlines is None:
            if n_sightlines is None:
                raise ValueError('Provide n_sightlines!')
            if redshift is None:
                raise ValueError('Provide redshift!')
        else:
            if n_sightlines is not None or redshift is not None:
                print('Using provided sightlines, ignoring other arguments!')
            n_sightlines = len(sightlines)
            redshift = max([sl.target_redshift for sl in sightlines])

        if reduce_sightlines and not find_halos:
            print("Can't reduce sightlines without halo assignment, so they won't be reduced!")

        # -- Select function to calculate and corresponding data fields needed -- #
        func,fields = self._choose_function(functype)
        
        # -- Check for saved sightlines in save_path, or generate and partition sightlines -- #
        if save_path is not None and sightlines is None:
            sightlines = self.load_sightlines(save_path)    # load sightlines

        if sightlines is not None:      # if sightlines were loaded / offered
            for sl in sightlines:
                sl.extend(self.sim,redshift)    # check to see if desired redshift longer than loaded, and extend whilst saving loaded data
            if (n_sightlines > len(sightlines)) & (method=='random'):   # if more sightlines wanted, extend list of sightlines
                sightlines.extend(self._generate_and_partition_sightlines(n_sightlines-len(sightlines),redshift,parallel_slgen,method,origin))
            elif all(sl.subsightline_reached(grid=True,halos=find_halos) == sl.num_sub_sightlines for sl in sightlines):   # if all sightlines are completely full, return sightlines
                print('Sightlines already processed!')
                if delete_data:
                    return sightlines
                else:
                    return sightlines, None, None
        else:  # if sightlines not created, create
            sightlines = self._generate_and_partition_sightlines(n_sightlines,redshift,parallel_slgen,method,origin)


        # -- Iterate over all snapshots required to traverse chosen redshift -- #
        snaps_required = len(np.unique(sightlines[0].sub_BoxRedshifts))
        start_snap = min([
            sl.sub_Snapshots[min(sl.subsightline_reached(grid=True, halos=find_halos), len(sl.sub_Snapshots) - 1)]
            for sl in sightlines
        ])
        for snap in range(start_snap,snaps_required):
            
            # -- Check snap completion -- #
            ptfind_check = snap < min([sl.sub_Snapshots[sl.subsightline_reached(grid=False)-1] for sl in sightlines])
            compute_check = snap < min([sl.sub_Snapshots[sl.subsightline_reached(grid=True)-1] for sl in sightlines])
            final_check = all(sl.subsightline_reached(grid=True) == sl.num_sub_sightlines for sl in sightlines) # passes if everything is complete

            if final_check:     
                continue

            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)
        
            trueSnapNum = self.sim._get_snap_num(snap)

            point_find_data = None
            if not ptfind_check:

                # -- Load data -- #
                point_find_data = self.sim.load_data(particle_type='gas',fields=self.sim.point_find_fields,snapNum=trueSnapNum,method=load_method)

                # -- Point finding architecture -- #
                architecture, radii, coarse_radius, giant_idx, giant_pts, giant_radii = self._finder_architecture(point_find_data,findtype)

                # -- Allocate point idx to each sub sightline -- #
                self._snapshot_points_in_sightlines(sightlines,snap,architecture,radii,coarse_radius,findtype,
                                                    giant_idx,giant_pts,giant_radii,parallel_findpts)
                
                if delete_data:
                    del(architecture)

                Cleanup_Memory()
            
            if (snap == 0) & (plot_sightlines):
                self.Vis.plot_many_sightlines(sightlines,n_sightlines=min(n_sightlines,20),points=point_find_data['Coordinates'],n_subsightlines=1)

            if not compute_check:
                
                fields = fields + self.sim.point_find_fields if point_find_data is None else fields
                data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum,method=load_method)
                data = data if point_find_data is None else data | point_find_data

                # -- Compute function for each sightline -- #
                self._snapshot_compute_sightlines(sightlines,data,func,
                                                snap,parallel_compute)
                
                if delete_data:
                    del(data)

                Cleanup_Memory()

            # -- Find halos in sightlines -- #
            ran_halos = False
            if find_halos:
                ran_halos = self.find_halos_in_sightlines(sightlines,parallel=parallel_halos,single_snap=snap,announce=False)
                self.assign_sightline_to_halos(sightlines)

                # -- Reduce Sightlines -- #
                if reduce_sightlines and (not compute_check or ran_halos):
                    self._snapshot_reduce_sightlines(sightlines,save_path if save_pointsidx else None,parallel_reduce)

            # -- Save sightlines -- #
            if save_path is not None and (not ptfind_check or not compute_check or ran_halos):
                self.save_sightlines(sightlines,save_path)

        if delete_data:
            return np.array(sightlines)
        else:
            return np.array(sightlines), data, architecture 



    # ------------- Post computation ------------- #

    def assign_sightline_to_halos(self,sightlines,method='sphere'):


        if method == 'sphere':
            for sl in _Smart_Tqdm(sightlines,desc='    assigning halo contribution using spherical approx'):
                sl.assign_to_halos()
        
        elif method == 'particles':
            

            snaps_required = len(np.unique(sightlines[0].sub_BoxRedshifts))

            print(f'Using halo particle membership (snapshots required: {snaps_required})',flush=True)

            for snap in range(snaps_required):
                print('\n',flush=True)
                print(f'------Snapshot {snap}------',flush=True)

                trueSnapNum = self.sim._get_snap_num(snap)

                # -- Load data -- #
                msg = f"    loading {self.sim.name} snapshot {trueSnapNum} data"
                print(msg,end='\r',flush=True)
                ts = clock()
                data = self.sim.load_data(particle_type='gas',fields=['ParticleIDs'],snapNum=trueSnapNum)
                _Progress_Print(msg,ts)

                for sl in _Smart_Tqdm(sightlines,desc='    assigning halo contribution'):
                    sl.assign_to_halos(method=method,particle_ids = data['ParticleIDs'], snapshot=snap,sim=self.sim)
                    


    def observe_halos_in_sightlines(self,sightlines,grid_path,filters=['lsst_g','lsst_r','lsst_i','lsst_z'],
                                    num_snaps=None, single_snap=None,parallel=False,save_interval=5,save_path=None):

        from ._galfinder_class import GalaxyFinder, load_grids_and_interps
        os.environ["SPS_HOME"] = self.sim.fsps_path
        import fsps

        filter_cache = {}
        for band in filters:
            wave_filt, trans = fsps.get_filter(band).transmission
            filter_cache[band] = (np.asarray(wave_filt), np.asarray(trans))

        if single_snap is not None:
            start_snap = single_snap
            snaps_required = single_snap + 1
        else:
            snaps_required = min(
                v for v in [sl.sub_Snapshots[sl.subsightline_reached(grid=False, halos=True) - 1] + 1 for sl in sightlines] + [num_snaps]
                if v is not None
            )
            start_snap = min([sl.sub_Snapshots[sl.subsightline_reached(grid=False, observed=True)] for sl in sightlines])

        for ii,snap in enumerate(range(start_snap,snaps_required)):
            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)

            galfinder = GalaxyFinder(snap, self.sim)
            interp_cache = load_grids_and_interps(grid_path,self.sim.redshifts[snap])

            if not _Is_Interactive():
                msg = f"    observing halos in sightlines"
                ts = clock()
                print(msg,end='\r',flush=True)

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                    delayed(sl.observe_halos)(galfinder,interp_cache=interp_cache,filter_cache=filter_cache) 
                    for sl in _Smart_Tqdm(sightlines,desc=f'    observing halos in sightlines [snap {snap}]')
                    )
            else:
                for sl in _Smart_Tqdm(sightlines, desc=f'    observing halos in sightline [snap {snap}]'):
                    sl.observe_halos(galfinder,interp_cache=interp_cache,filter_cache=filter_cache)

            if not _Is_Interactive():
                _Progress_Print(msg,ts)
            
            Cleanup_Memory()

            if save_path is not None:
                if (ii+1)%save_interval == 0 or snap == snaps_required - 1:
                    self.save_sightlines(sightlines,save_path)            


    def infer_halos_in_sightlines(self,sightlines,parallel=False,
                                  redshift_mode='truth',kcorrect_mode='kcorrect',m2l_mode='roediger15',halomass_mode='dpowerlaw_fit'):

        from ._inference_class import Inference

        def get_filters(sightlines):
            for sl in sightlines:
                for sh in sl.sub_Halos:
                    if sh != [None]:
                        for halo in sh:
                            if len(halo['ObservedGalaxies']) > 0:
                                return list(halo['ObservedGalaxies'][0]['VisiblePerFilter'].keys())
            return None  

        filters = get_filters(sightlines)
        
        inference = Inference(self.sim,filters=filters,load_kcorrect=True,
                              redshift_mode=redshift_mode,kcorrect_mode=kcorrect_mode,m2l_mode=m2l_mode,halomass_mode=halomass_mode)
        
        inference.process_redshifts(sightlines,filters)

        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(sl.infer_halos)(inference,filters) for sl in _Smart_Tqdm(sightlines,desc=f'Inferring halos in sightlines')
                )
        else:
            for sl in _Smart_Tqdm(sightlines, desc=f'Inferring halos in sightlines'):
                sl.infer_halos(inference,filters)


    def model_sightlines(self,sightlines,parallel=False,halo_params='inferred',igm_background='smooth_truth',density_smooth_mode='linear',density_smooth_kernel=1000,reduce=None):

        from ._inference_class import Inference

        filters = None
        if halo_params == 'inferred':

            def get_filters(sightlines):
                for sl in sightlines:
                    for sh in sl.sub_Halos:
                        if sh != [None]:
                            for halo in sh:
                                if 'GalaxyVisiblePerFilter' in halo:
                                    return list(halo['GalaxyVisiblePerFilter'].keys())
                return None  

            filters = get_filters(sightlines)

        inference = Inference(self.sim,halo_params=halo_params,igm_background=igm_background,density_smooth_mode=density_smooth_mode,density_smooth_kernel=density_smooth_kernel)


        if not _Is_Interactive():
            msg = f"Modelling sightlines"
            ts = clock()
            print(msg,end='\r',flush=True)

        if parallel:

            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'

            new_sightlines = Parallel(n_jobs=self.num_cores, backend='loky')(
                delayed(sl.model_sightline)(inference,filters,verbose=False,reduce=reduce) for sl in _Smart_Tqdm(sightlines,desc='Modelling sightlines')
                )
            
            for i,sl in enumerate(sightlines):
                sl.modelled = new_sightlines[i].modelled 

        else:
            for sl in _Smart_Tqdm(sightlines, desc='Modelling sightlines'):
                sl.model_sightline(inference,filters,verbose=False,reduce=reduce)

        if igm_background == 'smooth_truth':
            inference._fit_and_rescale_smooth_igm_density(sightlines)

        if not _Is_Interactive():
            _Progress_Print(msg,ts)


    def filter_sightlines(self,sightlines,observed=False,redshift=None,dvec_thresh=None,
                          min_halo_effmass=None,max_halo_effmass=None,
                          min_halo_effip=None,max_halo_effip=None,
                          min_halo_purity=None,max_halo_purity=None,
                        #   min_halo_gasfrac=0,max_halo_gasfrac=1,
                          min_num_halos=None,max_num_halos=None):
        
        mask = np.ones(len(sightlines), dtype=bool)

        if dvec_thresh is not None:
            print('Calculating Direction Vector Filter')
            l_max = np.float32(self.sim.cosmo.comoving_distance(
                redshift if redshift is not None else sightlines[0].target_redshift
            ).value * 1000)

            # How many periodic box replicas we need in each direction to cover
            # a sphere of radius l_max, plus a small safety margin
            n_max = int(np.ceil(l_max / self.sim.box_size)) + 2

            # Build the lattice of periodic image offsets: v = ..., -2L, -1L, 0, 1L, 2L, ...
            # then take the full 3D outer product to get every combination (i*L, j*L, k*L)
            v = np.arange(-n_max, n_max + 1, dtype=np.float32) * self.sim.box_size
            ii, jj, kk = np.meshgrid(v, v, v, indexing="ij")
            grid = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

            # Squared distance of each lattice/image point from the origin (the ray's start)
            grid_sq = np.sum(grid ** 2, axis=1)

            # Drop the origin itself (a ray trivially "intersects" its own starting box copy,
            # that's not a meaningful self-intersection) and drop any image points farther
            # than l_max away (they're outside the ray's integration length, irrelevant)
            keep = np.any(grid != 0, axis=1) & (grid_sq <= l_max ** 2)
            grid, grid_sq = grid[keep], grid_sq[keep]

            # --- Precision fix: promote the geometry arrays to float64 here ---
            # grid_sq and (later) proj**2 are both ~O(l_max^2), which can be ~1e13
            # in these units. The quantity we actually care about, pd2 = grid_sq - proj^2,
            # is only ~O(eps_min^2) — many orders of magnitude smaller than the two
            # numbers being subtracted. In float32 (~7 sig figs) that subtraction loses
            # essentially all precision exactly where it matters (near-zero pd2, i.e.
            # near self-intersections). float64 (~16 sig figs) keeps the residual clean.
            grid = grid.astype(np.float64)
            grid_sq = grid_sq.astype(np.float64)

            dvecs = np.array([sl.direction_vector for sl in sightlines], dtype=np.float64)
            lengths = np.array([sl.length for sl in sightlines], dtype=np.float64)

            eps_min = np.full(len(sightlines), np.inf, dtype=np.float64)

            chunk = 512

            # Process sightlines in batches to limit peak memory (grid × chunk arrays)
            for start in range(0, len(sightlines), chunk):
                nh      = dvecs[start : start + chunk]       # (B, 3) direction vectors in this batch
                l_chunk = lengths[start : start + chunk]     # (B,) integration lengths in this batch

                
                proj = grid @ nh.T   # Project every grid/image point onto each ray direction:

                # Only image points that project to a positive lambda within the ray's
                # actual length count — the ray doesn't exist before lambda=0 or after L_max.
                # 1e-6 avoids flagging the origin/near-origin numerically as "in range".
                in_range = (proj > 1e-6) & (proj <= l_chunk[None, :])

                # Perpendicular distance squared via Pythagoras:
                # |l_j|^2 = proj^2 + pd^2  =>  pd^2 = |l_j|^2 - proj^2
                pd2 = grid_sq[:, None] - proj ** 2

                # Guard against tiny negative values from floating point roundoff
                # (should be much rarer/smaller now that we're in float64)
                np.maximum(pd2, 0.0, out=pd2)

                # Image points outside the valid lambda range don't count as
                # potential self-intersections — push them to +inf so they never win the min
                pd2[~in_range] = np.inf

                # For each sightline in this batch, the closest approach distance
                # over all periodic images is eps_min
                best = pd2.min(axis=0)
                eps_min[start : start + chunk] = np.sqrt(best)

            # Normalize by box size (as in the paper) and keep only sightlines whose
            # closest self-approach is comfortably far relative to the box, i.e.
            # "clean" enough sightlines per the requested dvec_score threshold
            mask &= eps_min >= dvec_thresh

        where = np.where(mask)[0]
        for i,sl in enumerate(tqdm(sightlines[mask],desc='Filtering sightlines')):
            mask[where[i]] = sl.filter(cosmo=self.sim.cosmo,redshift=redshift,observed=observed,
                                       min_halo_effmass=min_halo_effmass,max_halo_effmass=max_halo_effmass,
                                       min_halo_effip=min_halo_effip,max_halo_effip=max_halo_effip,
                                       min_halo_purity=min_halo_purity,max_halo_purity=max_halo_purity,
                                    #    min_halo_gasfrac=min_halo_gasfrac,max_halo_gasfrac=max_halo_gasfrac,
                                       min_num_halos=min_num_halos,max_num_halos=max_num_halos)
            
        return mask
    




                                # def run_mcmc(self,sightlines,priors,redshift=None,nwalkers=32, nsteps=4000, burnin=500,initial_guess=None, seed=None):

                                #     import emcee
                                #     from ._inference_class import Inference
                                #     from tqdm import tqdm

                                #     if min([sl.subsightline_reached(modelled=True) for sl in sightlines]) == 0:
                                #         raise ValueError('Sightlines not fully modelled!')
                                #     if redshift is None:
                                #         redshift = sightlines[0].target_redshift
                                #     redshift = np.atleast_1d(redshift)

                                #     inference = Inference(self.sim)
                                #     inference.model_params = sightlines[0].modelled.model_params

                                #     sigma_igm = inference.build_sigma_igm_of_z(sightlines, self.sim.cosmo, redshift)
                                #     sigma_halo = inference.build_sigma_halo_of_z(sightlines, self.sim.cosmo, redshift, f_gas_ref=self.sim.f_gas)
                                #     sigma_model = np.sqrt(sigma_igm**2 + sigma_halo**2)
                                #     print(f'sigma_igm = {sigma_igm}, sigma_halo = {sigma_halo}, sigma_model = {sigma_model}')

                                #     print('\n')
                                #     dm_cgm_unit = np.array([sl.extract_compute(self.sim.cosmo, redshift=redshift, environment='CGM',
                                #                                             modelled=True, fgas=1.0, figm=1.0) for sl in tqdm(sightlines,desc='calculating fgas = 1.0 CGM DM')])
                                #     dm_igm_unit = np.array([sl.extract_compute(self.sim.cosmo, redshift=redshift, environment='IGM',
                                #                                             modelled=True, fgas=1.0, figm=1.0) for sl in tqdm(sightlines,desc='calculating figm = 1.0 IGM DM')])
                                #     dm_total_true = np.array([sl.extract_compute(self.sim.cosmo, redshift=redshift, environment='Total',
                                #                                             modelled=False) for sl in tqdm(sightlines,desc='calculating truth DM')])
                                #     print('\n')

                                #     param_names = list(priors.keys())
                                #     ndim = len(param_names)

                                #     if initial_guess is None:
                                #         initial_guess = tuple(
                                #             0.5 * (lo + hi) for lo, hi in priors.values()
                                #         )

                                #     rng = np.random.default_rng(seed)
                                #     pos = np.array(initial_guess) + 1e-2 * rng.standard_normal((nwalkers, ndim))
                                #     # clip into prior bounds
                                #     for j, (lo, hi) in enumerate(priors.values()):
                                #         pos[:, j] = np.clip(pos[:, j], lo + 1e-6, hi - 1e-6)

                                #     sampler = emcee.EnsembleSampler(
                                #         nwalkers, ndim, inference.log_probability,
                                #         args=(dm_cgm_unit, dm_igm_unit, dm_total_true, priors, sigma_model)
                                #     )
                                #     sampler.run_mcmc(pos, nsteps, progress=True)
                            
                                #     flat_samples = sampler.get_chain(discard=burnin, thin=10, flat=True)
                            
                                #     return sampler, flat_samples, param_names


    # def run_mcmc(self, sightlines, redshift, nwalkers=32, nsteps=4000,
    #             initial_guess=None, seed=None, prior_range=(0.0, 1.0),
    #             filt=None, sweep_param=None, sweep_values=None):

    #     import emcee
    #     from ._inference_class import Inference
    #     from tqdm import tqdm

    #     z_vals = np.atleast_1d(redshift)

    #     # -- blanket filter, applied once, before anything else -- #
    #     filt = filt or {}
    #     base_mask = self.filter_sightlines(sightlines, redshift=z_vals.max(), **filt)
    #     base_sightlines = sightlines[base_mask]

    #     if min([sl.subsightline_reached(modelled=True) for sl in base_sightlines]) == 0:
    #         raise ValueError('Sightlines not fully modelled!')

    #     inference = Inference(self.sim)
    #     inference.model_params = base_sightlines[0].modelled.model_params

    #     # -- precompute unit fields ONCE, on the base-filtered set -- #
    #     dm_cgm_unit = np.array([
    #         sl.extract_compute(self.sim.cosmo, redshift=z_vals, environment='CGM',
    #                             modelled=True, fgas=1.0)
    #         for sl in tqdm(base_sightlines, desc='calculating fgas=1.0 CGM DM')
    #     ])
    #     dm_igm_unit = np.array([
    #         sl.extract_compute(self.sim.cosmo, redshift=z_vals, environment='Total',
    #                             modelled=True, fgas=0.0)
    #         for sl in tqdm(base_sightlines, desc='calculating IGM DM')
    #     ])
    #     dm_total_true = np.array([
    #         sl.extract_compute(self.sim.cosmo, redshift=z_vals, environment='Total',
    #                             modelled=False)
    #         for sl in tqdm(base_sightlines, desc='calculating truth DM')
    #     ])

    #     sweeping = sweep_param is not None
    #     sweep_vals = sweep_values if sweeping else [None]

    #     results = []

    #     for sweep_val in sweep_vals:

    #         if sweeping:
    #             sub_mask = self.filter_sightlines(base_sightlines, redshift=z_vals.max(), **sweep_val)
    #         else:
    #             sub_mask = np.ones(len(base_sightlines), dtype=bool)

    #         sub_sightlines = base_sightlines[sub_mask]
    #         n_selected = sub_mask.sum()

    #         if n_selected < 20:
    #             print(f"sweep {sweep_param}={sweep_val}: only {n_selected} sightlines, skipping")
    #             results.append(None)
    #             continue

    #         # -- slice the already-precomputed arrays, no recomputation -- #
    #         dm_cgm_unit_sub = dm_cgm_unit[sub_mask]
    #         dm_igm_unit_sub = dm_igm_unit[sub_mask]
    #         dm_total_true_sub = dm_total_true[sub_mask]

    #         # -- sigma still needs to be built on this specific subsample -- #
    #         sigma_igm = inference.build_sigma_igm_of_z(sub_sightlines, self.sim.cosmo, z_vals)
    #         sigma_halo = inference.build_sigma_halo_of_z(sub_sightlines, self.sim.cosmo, z_vals,
    #                                                     f_gas_ref=self.sim.f_gas)
    #         sigma_per_z = np.sqrt(sigma_igm**2 + sigma_halo**2)

    #         n_z = len(z_vals)
    #         X_per_z, y_per_z, priors = [], [], {}
    #         for k in range(n_z):
    #             X_per_z.append(np.column_stack([dm_cgm_unit_sub[:, k], dm_igm_unit_sub[:, k]]))
    #             y_per_z.append(dm_total_true_sub[:, k])
    #             priors[f'f_gas_z{z_vals[k]:.2f}'] = prior_range
    #             priors[f'f_igm_z{z_vals[k]:.2f}'] = prior_range

    #         param_names = list(priors.keys())
    #         ndim = len(param_names)

    #         if initial_guess is None:
    #             init = tuple(0.5 * (lo + hi) for lo, hi in priors.values())
    #         else:
    #             init = initial_guess

    #         rng = np.random.default_rng(seed)
    #         nwalkers_eff = max(nwalkers, 4 * ndim)
    #         pos = np.array(init) + 1e-2 * rng.standard_normal((nwalkers_eff, ndim))
    #         for j, (lo, hi) in enumerate(priors.values()):
    #             pos[:, j] = np.clip(pos[:, j], lo + 1e-6, hi - 1e-6)

    #         sampler = emcee.EnsembleSampler(
    #             nwalkers_eff, ndim, inference.log_probability,
    #             args=(X_per_z, y_per_z, priors, sigma_per_z)
    #         )
    #         sampler.run_mcmc(pos, nsteps, progress=True)

    #         results.append({
    #             'sweep_value': sweep_val,
    #             'sampler': sampler,
    #             'param_names': param_names,
    #             'z_vals': z_vals,
    #             'n_selected': n_selected,
    #         })

    #     return results if sweeping else results[0]

    def run_mcmc(self, sightlines, redshift, mass_anchors, nwalkers=32, nsteps=4000,
            initial_guess=None, seed=None, fgas_prior=(0.0, 1.0),
            figm_prior=(0.0, 1.0), filt=None,mode='log'):

        import emcee
        from self._inference_class import Inference

        z_val = redshift #np.atleast_1d(redshift)
                            # if len(z_val) > 1:
                            #     raise ValueError("run_mcmc currently supports a single redshift only.")

        anchor_logM = np.log10(np.atleast_1d(mass_anchors))     # mass anchors for the mean fgas(M) line

        # -- Apply initial filter -- #
        if filt is not None:
            base_mask = self.filter_sightlines(sightlines, redshift=z_val, **filt)
            base_sightlines = sightlines[base_mask]
            print('\n')
        else:
            base_sightlines = sightlines
        n_sightlines = len(base_sightlines)

        if min([sl.subsightline_reached(modelled=True) for sl in base_sightlines]) == 0:
            raise ValueError('Sightlines not modelled!')

        inference = Inference(self.sim)

        print('Extracting Model and Truth DM')

        # -- Build halo information -- #
        halo_logM, dm_halo_unit, sl_index = inference.build_halo_arrays(      # length of each array is nhalos traversed by all sightlines
            base_sightlines, z_val
        )

        # -- Pre compute the per-sightline weights : w_i = sqrt(sum_j(u_j**2))  where u_j are the unit halo DMs per halo j in sightline i -- #
        halo_unit_sq_total = np.zeros(n_sightlines)
        np.add.at(halo_unit_sq_total, sl_index, dm_halo_unit**2)
        halo_unit_rss = np.sqrt(halo_unit_sq_total)

        # -- IGM DM scaled to figm = 1 -- #
        dm_igm_unit = np.array([
            sl.extract_compute(self.sim.cosmo, redshift=z_val, environment='IGM',
                                modelled=True, figm=1.0)
            for sl in tqdm(base_sightlines, desc='    extracting unit IGM DM')
        ])

        # -- Total "measured" values -- #
        dm_total_true = np.array([
            sl.extract_compute(self.sim.cosmo, redshift=z_val, environment='Total',
                                modelled=False)
            for sl in tqdm(base_sightlines, desc='    extracting truth DM')
        ])

        print('\n')
        print('Generating Sigmas')

        # -- Build a sigma for the igm term by comparing the model to the truth -- #
        sigma_igm = inference.build_sigma_igm(base_sightlines, self.sim.cosmo, z_val, mode)
                # sigma_halo = inference.build_sigma_halo(base_sightlines, self.sim.cosmo, z_val, mode)
                # sigma = np.sqrt(sigma_igm**2 + sigma_halo**2)
        sigma = sigma_igm
        print('\n')

        # -- Prior range and initial guesses per variable -- #
        priors = {f'f_gas_M{m:.2f}': fgas_prior for m in anchor_logM}
        priors['f_igm'] = figm_prior
        priors['sigma_fgas'] = (0.01, 0.5)

        param_names = list(priors.keys())
        ndim = len(param_names)

        if initial_guess is None:
            init = tuple(0.5 * (lo + hi) for lo, hi in priors.values())
        else:
            init = initial_guess

        # -- Randomize starting walkers -- #
        rng = np.random.default_rng(seed)
        nwalkers_eff = max(nwalkers, 4 * ndim)
        pos = np.array(init) + 1e-2 * rng.standard_normal((nwalkers_eff, ndim))
        for j, (lo, hi) in enumerate(priors.values()):
            pos[:, j] = np.clip(pos[:, j], lo + 1e-6, hi - 1e-6)

        
        print('Running emcee')

        sampler = emcee.EnsembleSampler(
            nwalkers_eff, ndim, inference.log_probability,
            args=(anchor_logM, halo_logM, dm_halo_unit, sl_index, n_sightlines,
                dm_igm_unit, dm_total_true, sigma, halo_unit_rss, priors,mode)
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

        return {
            'sampler': sampler,
            'param_names': param_names,
            'anchor_logM': anchor_logM,
            'z_val': z_val,
            'n_selected': n_sightlines,
        }