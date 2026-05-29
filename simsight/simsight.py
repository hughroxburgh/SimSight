from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

from time import time
import numpy as np

from .sims import load_sim
from ._visualiser_class import VisualSim

class SightlineSim():

    def __init__(self,data_path,parallel=False,num_cores=None,backend='loky',fsps_path=None):
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
        self.sim = load_sim(data_path,fsps_path)

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
                print(f'N HEALpix sightlines = {n_sightlines}, origin = {origin}')

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
                delayed(sl.partition)(self.sim,verbose=False) for sl in tqdm(sightlines[1:],desc='Creating all sightlines')
                )
        else:

            for sl in tqdm(sightlines[1:],desc='Creating all sightlines'):
                sl.partition(self.sim,verbose=False)

        return sightlines
    


    # ------------- Find points / halos in given sightlines ------------- #

    def _snapshot_points_in_sightlines(self,sightlines,snapshot,tree,radii,coarse_radius,giant_idx,giant_pts,giant_radii,parallel=False):

        from ._point_find import Points_In_Sightline
        
        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(Points_In_Sightline)(sl, snapshot, tree,radii,coarse_radius,giant_idx,giant_pts,giant_radii,treebased=True)
                for sl in tqdm(sightlines, desc="    finding points in sightlines")
            )
        else:
            for sl in tqdm(sightlines, desc="    finding points in sightlines"):
                Points_In_Sightline(sl,snapshot,tree,radii,coarse_radius,giant_idx,giant_pts,giant_radii,treebased=True)


    def find_halos_in_sightlines(self,sightlines,parallel=False,snaps=None):

        from ._point_find import Halos_In_Sightline

        snaps_required = min(v for v in [sightlines[0].sub_Snapshots[-1]+1, snaps] if v is not None)

        for snap in range(snaps_required):

            trueSnapNum = self.sim._get_snap_num(snap)
            halos = self.sim.load_halos(trueSnapNum)

            radii = np.array([h['Radius'] for h in halos], dtype=np.float32)
            com = np.array([h['Pos'] for h in halos], dtype=np.float32)

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend='loky')(
                    delayed(Halos_In_Sightline)(sl, snap, halos,com,radii) for sl in tqdm(sightlines,desc='Finding halos in sightlines'))
            else:
                for sightline in tqdm(sightlines, desc='Finding halos in sightlines'):
                    Halos_In_Sightline(sightline,snap,halos,com,radii)

            print('\n')

            


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
                for file in tqdm(files[:n_files],desc='Loading sightlines'):
                    with open(file,'rb') as f:
                        sightlines.append(pickle.load(f))
                return sightlines
        else:
            print('No sightlines saved in save_path.')
            return None
                    

    def save_sightlines(self,sightlines,save_path):

        for i in tqdm(range(len(sightlines)), desc='    saving sightlines'):
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
                for sl in tqdm(sightlines, desc="    computing snapshot sightlines")
            )
        else:
            results = [Compute_Sightline(sl, self.sim, data, func, snapshot) for sl in tqdm(sightlines, desc="    computing snapshot sightlines")]

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
            print('\n')
            print(f'------Snapshot {snap}------')
        
            trueSnapNum = self.sim._get_snap_num(snap)

            # -- Load data -- #
            print(f"    loading {self.sim.name} snapshot {trueSnapNum} data",end='\r')
            ts = time()
            data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum)
            print(f"    loading {self.sim.name} snapshot {trueSnapNum} data -- Done ({time()-ts:.1f}s)")

            # -- Define particle radii and the maximum radius to search within -- #
            radii = self.sim.radius_mapping(data)
            coarse_radius = np.percentile(radii,99.9)

            Points_In_Sightline(sightline,snap,data['Coordinates'],radii,coarse_radius,treebased=False)

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


    def run_many_sightlines(self,n_sightlines,redshift,method='random',origin=None,functype='DM',
                            parallel_slgen=False,parallel_findpts=False,parallel_compute=False,
                            delete_data=True,save_path=None,plot_sightlines=False):
        """
        Run full loop over chosen number of sightlines.
        """

        from scipy.spatial import cKDTree

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

            print('\n')
            print(f'------Snapshot {snap}------')
        
            trueSnapNum = self.sim._get_snap_num(snap)

            # -- Load data -- #
            print(f"    loading {self.sim.name} snapshot {trueSnapNum} data",end='\r')
            ts = time()
            data = self.sim.load_data(particle_type='gas',fields=fields,snapNum=trueSnapNum)
            print(f"    loading {self.sim.name} snapshot {trueSnapNum} data -- Done ({time()-ts:.1f}s)")

            # -- Define particle radii and the maximum radius to search within -- #
            radii = self.sim.radius_mapping(data)
            coarse_radius = np.percentile(radii,99.9)
            giant_idx = np.where(radii > coarse_radius)[0]
            giant_pts = data['Coordinates'][giant_idx]
            giant_radii = radii[giant_idx]

                        # rEffs = (3/(4*np.pi)*data['Masses']/data['Density'])**(1/3)
                        # del(data['Masses'])

            #  -- Generate KDTree of all points in simulation -- #
            print(f"    generating KDTree",end='\r')
            ts = time()
            tree = cKDTree(data['Coordinates'])
            print(f"    generating KDTree -- Done ({time()-ts:.1f}s)")

            # -- Allocate point idx to each sub sightline -- #
            self._snapshot_points_in_sightlines(sightlines,snap,tree,radii,coarse_radius,giant_idx,giant_pts,giant_radii,parallel_findpts)

            if delete_data:
                del(tree)
            
            if (snap == 0) & (plot_sightlines):
                self.Vis.plot_many_sightlines(sightlines,n_sightlines=min(n_sightlines,20),points=data['Coordinates'],n_subsightlines=1)

            self._snapshot_compute_sightlines(sightlines,data,func,snap,parallel_compute)

            if save_path is not None:
                self.save_sightlines(sightlines,save_path)

            if delete_data:
                del(data)

        if delete_data:
            return sightlines
        else:
            return sightlines, data, tree 


    # ------------- Post computation ------------- #

    def assign_sightline_to_halos(self,sightlines,method='sphere'):


        if method == 'sphere':
            for sl in tqdm(sightlines,desc='Assigning halo contribution using spherical approx'):
                sl.assign_to_halos()
        
        elif method == 'particles':
            

            snaps_required = len(np.unique(sightlines[0].sub_BoxRedshifts))

            print(f'Using halo particle membership (snapshots required: {snaps_required})')

            for snap in range(snaps_required):
                print('\n')
                print(f'------Snapshot {snap}------')

                trueSnapNum = self.sim._get_snap_num(snap)

                # -- Load data -- #
                print(f"    loading {self.sim.name} snapshot {trueSnapNum} data",end='\r')
                ts = time()
                data = self.sim.load_data(particle_type='gas',fields=['ParticleIDs'],snapNum=trueSnapNum)
                print(f"    loading {self.sim.name} snapshot {trueSnapNum} data -- Done ({time()-ts:.1f}s)")

                for sl in tqdm(sightlines,desc='Assigning halo contribution'):
                    sl.assign_to_halos(method=method,particle_ids = data['ParticleIDs'], snapshot=snap,sim=self.sim)



    def observe_halos_in_sightlines(self,sightlines,grid_path,filters=['lsst_g','lsst_r','lsst_i','lsst_z'],snaps=None,parallel=False):

        from ._galfinder_class import GalaxyFinder

        snaps_required = min(v for v in [sightlines[0].sub_Snapshots[sightlines[0].subsightline_reached(grid=False,halos=True)-1]+1, snaps] if v is not None)

        for snap in range(snaps_required):
            print('\n')
            print(f'------Snapshot {snap}------')

            galfinder = GalaxyFinder(snap, self.sim)

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                    delayed(sl.observe_halos)(galfinder,grid_path,filters) for sl in tqdm(sightlines,desc='Observing halos in sightlines')
                    )
            else:
                for sl in tqdm(sightlines, desc='Observing halos in sightlines'):
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
            print('\n')
            print(f'------Snapshot {snap}------')

            if parallel:
                sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                    delayed(sl.infer_halos)(inference,filters) for sl in tqdm(sightlines,desc='Inferring halos in sightlines')
                    )
            else:
                for sl in tqdm(sightlines, desc='Inferring halos in sightlines'):
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

        if parallel:
            sightlines = Parallel(n_jobs=self.num_cores, backend=self.backend)(
                delayed(sl.model_sightline)(inference,halo_params,igm_background,density_smooth_kernel,
                                            filters,verbose=False) for sl in tqdm(sightlines,desc='Modelling sightlines')
                )
        else:
            for sl in tqdm(sightlines, desc='Modelling sightlines'):
                sl.model_sightline(inference,halo_params,igm_background,density_smooth_kernel,filters,verbose=False)
