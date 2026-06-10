import multiprocessing
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
import sys

from time import time as clock
import numpy as np

from .sims import load_sim
from ._visualiser_class import VisualSim
from ._utils import _Progress_Print, _Smart_Tqdm, _Is_Interactive

class SightlineSim():

    def __init__(self,data_path,halo_path_structure=None,parallel=False,num_cores=None,backend='threading',fsps_path=None):
        """
        data_path : where the data is stored on the servers. As long as the words 'TNG' or 'SIMBA' are inside, the class will try to conform to those standards.
        parallel : run things in parallel -- SEEMS TO BE WORSE SO JUST LEAVE FALSE FOR NOW
        """

        # -- Set up init -- #
        self.data_path = data_path  
        self.parallel = parallel
        self.backend = backend
        if num_cores is None:
            self.num_cores = int(multiprocessing.cpu_count())
        else:
            self.num_cores = num_cores

        # -- Select Simulation Suite -- #
        self.sim = load_sim(data_path,halo_path_structure,fsps_path)

        # -- Visualiser -- #
        self.Vis = VisualSim(self.sim)

    
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
    


    # ------------- Find points / halos in given sightlines ------------- #

    def _snapshot_points_in_sightlines(self,sightlines,snapshot,architecture,radii,coarse_radius,findtype,
                                       giant_idx,giant_pts,giant_radii,parallel=False):

        from ._point_find import Points_In_Sightline
        
        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(Points_In_Sightline)(sl, snapshot, architecture,radii,coarse_radius,findtype,giant_idx,giant_pts,giant_radii)
                for sl in _Smart_Tqdm(sightlines, desc=f"    finding points in sightlines [snap {snapshot}]")
            )
        else:
            for sl in _Smart_Tqdm(sightlines, desc=f"    finding points in sightlines [snap {snapshot}]"):
                Points_In_Sightline(sl,snapshot,architecture,radii,coarse_radius,findtype,giant_idx,giant_pts,giant_radii)


    def find_halos_in_sightlines(self,sightlines,parallel=False,snaps=None):

        from ._point_find import Halos_In_Sightline

        snaps_required = min(v for v in [sightlines[0].sub_Snapshots[-1]+1, snaps] if v is not None)

        for snap in range(snaps_required):

            print(f'------Snapshot {snap}------',flush=True)

            # -- Load halos -- #
            if not _Is_Interactive():
                msg = f"    loading {self.sim.name} snapshot {trueSnapNum} halos"
                ts = clock()
                print(msg,end='\r',flush=True)

            trueSnapNum = self.sim._get_snap_num(snap)
            halos = self.sim.load_halos(trueSnapNum)

            if not _Is_Interactive():
                _Progress_Print(msg,ts)
            # --------------- #

            radii = np.array([h['Radius'] for h in halos], dtype=np.float32)
            com = np.array([h['Pos'] for h in halos], dtype=np.float32)

            # -- Find halos in sightlines -- #
            if not _Is_Interactive():
                msg = f"    finding halos in sightlines"
                ts = clock()
                print(msg,end='\r',flush=True)

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend='loky')(
                    delayed(Halos_In_Sightline)(sl, snap, halos,com,radii) for sl in _Smart_Tqdm(sightlines,desc='Finding halos in sightlines'))
            else:
                for sightline in _Smart_Tqdm(sightlines, desc='Finding halos in sightlines'):
                    Halos_In_Sightline(sightline,snap,halos,com,radii)

            if not _Is_Interactive():
                _Progress_Print(msg,ts)
            # ------------------------------ #

            print('\n',flush=True)

            


    # ------------- Loading / saving sightlines ------------- #

    def load_sightlines(self,save_path,percent=100):

        import pickle
        import os
        from glob import glob

        if os.path.exists(save_path):
            files = sorted(glob(f'{save_path}/*.pkl'))
            if len(files) > 0:
                n_files = int(percent*len(files)/100)
                sightlines = []
                for file in _Smart_Tqdm(files[:n_files],desc='Loading sightlines'):
                    with open(file,'rb') as f:
                        sightlines.append(pickle.load(f))
                return sightlines
        else:
            print('No sightlines saved in save_path.',flush=True)
            return None
                    

    def save_sightlines(self,sightlines,save_path):

        for i in _Smart_Tqdm(range(len(sightlines)), desc='    saving sightlines'):
            sightlines[i].save(save_path,i)



    # ------------- Runnning computation ------------- #

    def _choose_function(self,f):

        from ._compute import Calc_Ray_Density, Calc_Ray_DM

        mapping = {'Density': {'func': Calc_Ray_Density, 
                               'fields': ['Coordinates','Density','Masses','SmoothingLength']},

                    'DM' : {'func': Calc_Ray_DM, 
                            'fields': ['Coordinates','Density','Masses','SmoothingLength','StarFormationRate','ElectronAbundance']}}
        #       

        return mapping[f]['func'], mapping[f]['fields']
    

    def _snapshot_compute_sightlines(self, sightlines, data, func, snapshot,parallel=False):

        from ._compute import Compute_Sightline
        
        if parallel:
            results = Parallel(n_jobs=self.num_cores, backend='threading')(
                delayed(Compute_Sightline)(sl, self.sim, data, func, snapshot)
                for sl in _Smart_Tqdm(sightlines, desc=f"    computing snapshot sightlines [snap {snapshot}]")
            )
        else:
            results = [Compute_Sightline(sl, self.sim, data, func, snapshot) for sl in _Smart_Tqdm(sightlines, desc=f"    computing snapshot sightlines [snap {snapshot}]")]

        for sl, sl_results in zip(sightlines, results):
            for sub_idx, compute, density, lengths, ids in sl_results:
                sl.sub_Compute[sub_idx] = compute
                sl.sub_Density[sub_idx] = density
                sl.sub_Grid[sub_idx] = lengths
                sl.sub_Cells[sub_idx] = ids


            
    def run_single_sightline(self,redshift,origin=None,direction_vector=None,functype='DM',
                             delete_data=True,save_path=None,plot_sightline=False):
        
        from ._compute import Compute_Sightline
        from ._point_find import Points_In_Sightline
        
        # -- Select function to calculate and corresponding data fields needed -- #
        func,fields = self._choose_function(functype)
        
        sightline = None
        if save_path is not None:
            sightline = self.load_sightlines(save_path)[0]

        if sightline is None:    
            sightline = self.generate_sightlines(redshift,n_sightlines=1,origin=origin,direction_vector=direction_vector)
            sightline.partition(self.sim)

        snaps_required = len(np.unique(sightline.sub_BoxRedshifts))
        for idx,subPoints in enumerate(sightline.sub_PointsIdx):
            if len(subPoints) == 0:
                break
        start_snap = sightline.sub_Snapshots[idx]

        for snap in range(start_snap,snaps_required):
            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)
        
            trueSnapNum = self.sim._get_snap_num(snap)

            # -- Load data -- #
            msg = f"    loading {self.sim.name} snapshot {trueSnapNum} data"
            print(msg,end='\r',flush=True)
            ts = clock()
            data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum)
            _Progress_Print(msg,ts)

            # -- Define particle radii and the maximum radius to search within -- #
            radii = self.sim.radius_mapping(data)
            coarse_radius = np.percentile(radii,99.9)

            Points_In_Sightline(sightline,snap,data['Coordinates'],radii,coarse_radius,findtype='cylinder')

            if (snap == 0) & (plot_sightline):
                self.Vis.plot3d()
                self.Vis.plot_sightline(sightline,data['Coordinates'])

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

            return {'voxels':voxels,
                    'coords':data['Coordinates'],
                    'grid_size':grid_size}, radii, coarse_radius, giant_idx, giant_pts, giant_radii

    def run_many_sightlines(self,n_sightlines,redshift,method='random',origin=None,
                            functype='DM',findtype='tree',load_method='custom',
                            parallel_slgen=False,parallel_findpts=False,parallel_compute=False,
                            delete_data=True,save_path=None,plot_sightlines=False):
        """
        Run full loop over chosen number of sightlines.
        """

        # -- Select function to calculate and corresponding data fields needed -- #
        func,fields = self._choose_function(functype)
        
        # -- Check for saved sightlines in save_path, or generate and partition sightlines -- #
        sightlines = None
        if save_path is not None:
            sightlines = self.load_sightlines(save_path)    # load sightlines

            if sightlines is not None:      # if sightlines were loaded
                for sl in sightlines:
                    sl.extend(self.sim,redshift)    # check to see if desired redshift longer than loaded, and extend whilst saving loaded data

                if (n_sightlines > len(sightlines)) & (method=='random'):   # if more sightlines wanted, extend list of sightlines
                    sightlines.extend(self._generate_and_partition_sightlines(n_sightlines-len(sightlines),redshift,parallel_slgen,method,origin))

                elif len(sightlines[-1].sub_Compute[-1]) > 0:   # if all sightlines are completely full, return sightlines
                    print('Sightlines already processed!')
                    
                    if delete_data:
                        return sightlines
                    else:
                        return sightlines, None, None
                    
        if sightlines is None:  # if sightlines not created, create
            sightlines = self._generate_and_partition_sightlines(n_sightlines,redshift,parallel_slgen,method,origin)


        # -- Iterate over all snapshots required to traverse chosen redshift -- #
        snaps_required = len(np.unique(sightlines[0].sub_BoxRedshifts))
        for idx,subPoints in enumerate(sightlines[-1].sub_PointsIdx):
            if len(subPoints) == 0:
                break
        start_snap = sightlines[-1].sub_Snapshots[idx]

        for snap in range(start_snap,snaps_required):

            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)
        
            trueSnapNum = self.sim._get_snap_num(snap)

            # -- Load data -- #
            msg = f"    loading {self.sim.name} snapshot {trueSnapNum} data"
            print(msg,end='\r',flush=True)
            ts = clock()
            data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum,method=load_method)
            _Progress_Print(msg,ts)
            # --------------- #

            # -- Point finding architecture -- #
            architecture, radii, coarse_radius, giant_idx, giant_pts, giant_radii = self._finder_architecture(data,findtype)
            # -------------------------------- #


            # -- Allocate point idx to each sub sightline -- #
            if not _Is_Interactive():
                msg = f"    finding points in sightlines"
                ts = clock()
                print(msg,end='\r',flush=True)
            self._snapshot_points_in_sightlines(sightlines,snap,architecture,radii,coarse_radius,findtype,
                                                giant_idx,giant_pts,giant_radii,parallel_findpts)
            if not _Is_Interactive():
                _Progress_Print(msg,ts)
            # ---------------------------------------------- #
            
            if delete_data:
                del(architecture)
            
            if (snap == 0) & (plot_sightlines):
                self.Vis.plot_many_sightlines(sightlines,n_sightlines=min(n_sightlines,20),points=data['Coordinates'],n_subsightlines=1)

            # -- Compute function for each sightline -- #
            if not _Is_Interactive():
                msg = f"    computing snapshot sightlines"
                ts = clock()
                print(msg,end='\r',flush=True)
            self._snapshot_compute_sightlines(sightlines,data,func,
                                              snap,parallel_compute)
            if not _Is_Interactive():
                _Progress_Print(msg,ts)
            # ----------------------------------------- #

            # -- Save sightlines -- #
            if save_path is not None:
                if not _Is_Interactive():
                    msg = f"    saving sightlines to {save_path}"
                    ts = clock()
                    print(msg,end='\r',flush=True)
                self.save_sightlines(sightlines,save_path)
                if not _Is_Interactive():
                    _Progress_Print(msg,ts)
            # --------------------- #

            if delete_data:
                del(data)

        if delete_data:
            return sightlines
        else:
            return sightlines, data, architecture 

    # ------------- Post computation ------------- #

    def assign_sightline_to_halos(self,sightlines,method='sphere'):


        if method == 'sphere':
            for sl in _Smart_Tqdm(sightlines,desc='Assigning halo contribution using spherical approx'):
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

                for sl in _Smart_Tqdm(sightlines,desc='Assigning halo contribution'):
                    sl.assign_to_halos(method=method,particle_ids = data['ParticleIDs'], snapshot=snap,sim=self.sim)



    def observe_halos_in_sightlines(self,sightlines,grid_path,filters=['lsst_g','lsst_r','lsst_i','lsst_z'],snaps=None,parallel=False):

        from ._galfinder_class import GalaxyFinder

        snaps_required = min(v for v in [sightlines[0].sub_Snapshots[sightlines[0].subsightline_reached(grid=False,halos=True)-1]+1, snaps] if v is not None)

        for snap in range(snaps_required):
            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)

            galfinder = GalaxyFinder(snap, self.sim)

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                    delayed(sl.observe_halos)(galfinder,grid_path,filters) for sl in _Smart_Tqdm(sightlines,desc=f'Observing halos in sightlines [snap {snap}]')
                    )
            else:
                for sl in _Smart_Tqdm(sightlines, desc=f'Observing halos in sightline [snap {snap}]'):
                    sl.observe_halos(galfinder,grid_path,filters)

    def infer_halos_in_sightlines(self,sightlines,snaps=None,parallel=False):

        from ._inference_class import Inference

        snaps_required = min(v for v in [sightlines[0].sub_Snapshots[sightlines[0].subsightline_reached(grid=False,observed=True)-1]+1, snaps] if v is not None)

        def get_filters(sightlines):
            for sl in sightlines:
                for sh in sl.observed.sub_Halos:
                    if sh != [None]:
                        for halo in sh:
                            if 'GalaxyVisiblePerFilter' in halo:
                                return list(halo['GalaxyVisiblePerFilter'].keys())
            return None  

        filters = get_filters(sightlines)
        
        inference = Inference(self.sim,filters=filters,load_kcorrect=True)

        for snap in range(snaps_required):
            print('\n',flush=True)
            print(f'------Snapshot {snap}------',flush=True)

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                    delayed(sl.infer_halos)(inference,filters) for sl in _Smart_Tqdm(sightlines,desc=f'Inferring halos in sightlines [snap {snap}]')
                    )
            else:
                for sl in _Smart_Tqdm(sightlines, desc=f'Inferring halos in sightlines [snap {snap}]'):
                    sl.infer_halos(inference,filters)


    def model_sightlines(self,sightlines,parallel=False,halo_params='inferred',igm_background='smooth_truth',density_smooth_kernel=1000):

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

        inference = Inference(self.sim)


        if not _Is_Interactive():
            msg = f"Modelling sightlines"
            ts = clock()
            print(msg,end='\r',flush=True)

        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(sl.model_sightline)(inference,halo_params,igm_background,density_smooth_kernel,
                                            filters,verbose=False) for sl in _Smart_Tqdm(sightlines,desc='Modelling sightlines')
                )
        else:
            for sl in _Smart_Tqdm(sightlines, desc='Modelling sightlines'):
                sl.model_sightline(inference,halo_params,igm_background,density_smooth_kernel,filters,verbose=False)

        if not _Is_Interactive():
            _Progress_Print(msg,ts)