import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numba import njit
import os 
import pickle
from copy import deepcopy
from glob import glob

from scipy.spatial.distance import cdist

from astropy.cosmology import z_at_value
import astropy.units as u

from ._compute import Transform_Points
from ._utils import _Get_Colours

LIMITING_MAGS = {
    'lsst_g' : 25.0,
    'lsst_r' : 24.7,
    'lsst_i' : 24.0,
    'lsst_z' : 23.3
}

class Sightline():

    def __init__(self,target_redshift=None,length=None,origin=None,direction_vector=None,end_point=None,box_size=None,parent_sightline=True):
        """
        Sightline class!

        radius - radius of cylinder. In units of whatever the box you're working in is (ie. ckpc)
        targetZ - redshift of target distance away.
        length - length of sightline, is superceded by targetZ.
        origin - 3D origin of sightline.
        direction_vector - 3D direction vector of sighltine.
        end_point - specify end point, origin, and length, and direction_vector will be found to take you from A to B at roughly that length.     
        """

        # -- Validate inputs -- #
        self._validate_inputs(origin,target_redshift,length,end_point,direction_vector)

        # -- Always required parameter -- #
        self.origin = np.array(origin)

        # -- Other variables that may be set now -- #
        self.length = length
        self.target_redshift = target_redshift

        self.sightline_idx = 0

        # -- Calculate direction vector and find resulting vector basis -- #
        self._process_direction_vector(direction_vector,end_point,box_size)
        self._find_vector_basis() 

        if parent_sightline:
            # -- Preallocate variables -- #
            self.sub_Origins = None
            self.sub_Lengths = None
            self.sub_PointsIdx = None
            self.sub_BoxRedshifts = None
            self.sub_Snapshots = None
            self.sub_Density = None
            self.sub_Compute = None
            self.sub_Grid = None
            self.sub_Cells = None
            self.sub_Halos = None
            self.sub_CellConditions = None
            self.sub_HaloAssignment = None

            self.num_sub_sightlines = None      # number of sub sightlines

            self.sub_Observed = None
            self.modelled = None
            self.halo_inference_params = None

    def _validate_inputs(self,origin,target_redshift,length,end_point,direction_vector):
        """
        Validate inputs. 
        """

        if origin is None:
            e = print("Please specify origin!")
            raise ValueError(e)               

        if (target_redshift is not None) & (length is not None):
            e = print("Don't specify both redshift and length!")
            raise ValueError(e)     
        
        if (direction_vector is not None) & (end_point is not None): 
            e = print("Don't specify both direction vector and end point!")
            raise ValueError(e)

    def _process_direction_vector(self,direction_vector,end_point,box_size,num_boxes=100):
        """
        Used for calculating unit direction vector from input or choosing one if given an 'end_point' and desired 'length'
        """

        if direction_vector is not None:    # else, if given direction vector, normalise and use
            vector = np.array(direction_vector).astype(np.float32)
            self.direction_vector = vector / np.linalg.norm(vector)

        elif (end_point is not None) & (self.length is None):    # else, if given end_point with no desired length, choose most direct direction_vector
            vector = (np.array(end_point) - np.array(self.origin)).astype(np.float32)
            self.length = np.linalg.norm(vector).astype(np.float32)
            self.direction_vector = vector / self.length

        elif (end_point is not None):   # else, calculate direction vector that creates a ray with 'length' from origin to end_point. Checks closest loc out of 'num_boxes^3' attempts
            
            # generate same location in every box 
            xPoints = np.arange(end_point[0],end_point[0]+num_boxes*box_size,box_size)
            yPoints = np.arange(end_point[0],end_point[0]+num_boxes*box_size,box_size)
            zPoints = np.arange(end_point[0],end_point[0]+num_boxes*box_size,box_size)
            x,y,z = np.array(np.meshgrid(xPoints,yPoints,zPoints)).reshape(3,num_boxes**3)
            end_points = np.array(list(zip(x,y,z)))

            # calculate distance from origin and choose distance closest to 'length'
            d = cdist(np.array([self.origin]),end_points)
            true_end_point = end_points[np.argmin(abs(d-self.length))]  

            # set direction vector
            vector = true_end_point-self.origin
            self.length = np.linalg.norm(vector).astype(np.float32)
            self.direction_vector = (vector / self.length).astype(np.float32)


    def __str__(self):
        # Controls what shows up when you use print(obj)
        return f"Sightline(Origin={self.origin}\n\
          Target Redshift={self.target_redshift}\n\
          Direction={self.direction_vector}\n\
          Num SubSightlines={self.num_sub_sightlines})"
        
    def __repr__(self):
        # Controls what shows up when you just type obj in the interpreter
        return f"Sightline(Origin={self.origin}\n\
          Target Redshift={self.target_redshift}\n\
          Direction={self.direction_vector}\n\
          Num SubSightlines={self.num_sub_sightlines})"

    def sightline_distance(self,cosmo):
        """
        Adds a length based on input redshift for given sightline.
        """

        if self.target_redshift is not None:    # calculate comoving distance from redshift
            self.length = np.float32(cosmo.comoving_distance(self.target_redshift).value*1000)

        else:   # calculate redshift from comoving distance
            self.target_redshift = z_at_value(cosmo.comoving_distance, self.length*u.kpc)

    def _find_vector_basis(self):

        """
        Computes an orthonormal basis aligned with the direction vector.
        """
        v1 = np.asarray(self.direction_vector, dtype=np.float64)
        v1 /= np.linalg.norm(v1)

        # Choose a reference vector that is not colinear
        if abs(v1[2]) < 0.9:
            v2 = np.array([0.0, 0.0, 1.0])
        else:
            v2 = np.array([1.0, 0.0, 0.0])

        # Gram-Schmidt process to get orthogonal vectors
        v2 -= np.dot(v2, v1) * v1
        v2 /= np.linalg.norm(v2)

        v3 = np.cross(v1, v2)

        # Final basis and transformation matrix
        self.basis_vectors = np.stack((v1, v2, v3), axis=0).astype(np.float32)
        self.transformation_matrix = self.basis_vectors.T


    # ------------- Function for plotting ------------- #

    def gen_line(self,subsightline=None,n_samples=1000):
        """
        Generates a 3D line for the sightline.
        """

        if self.length is None:
            e = 'Sightline has no length!'
            raise AttributeError(e)
        
        if subsightline is None:
            origin = self.origin
            length = self.length
        else:
            origin = self.sub_Origins[subsightline]
            length = self.sub_Lengths[subsightline]

        line = np.linspace(origin,origin+self.direction_vector*length,n_samples).astype(np.float32)

        return line     



    # ------------- Partitioning full sightline into subsightlines ------------- #

    @staticmethod
    @njit(fastmath=True)
    def _find_exterior_rays(origin, length, direction_vector, box_size):
        """ 
        Finds origins and lengths PACMAN style for sightlines that would otherwise exit the box. 
        """

        max_lengths = np.empty(3, dtype=np.float64)

        # Compute distance to next boundary along each axis
        for i in range(3):
            if direction_vector[i] > 0:
                max_lengths[i] = (box_size - origin[i]) / direction_vector[i]
            elif direction_vector[i] < 0:
                max_lengths[i] = origin[i] / np.abs(direction_vector[i])
            else:
                # Ray parallel to this axis → infinite distance
                max_lengths[i] = 1e20

        # Find the first boundary hit
        clipped_length = np.min(max_lengths)
        if clipped_length > length:
            return None, None
        
        axis = np.argmin(max_lengths)  # axis of first intersection

        # Compute new origin after wrapping
        next_origin = origin + direction_vector * clipped_length


        # Wrap along the intersected axis
        if direction_vector[axis] > 0:
            next_origin[axis] = 0.0
        else:
            next_origin[axis] = box_size


        return next_origin, length-clipped_length
    
    @staticmethod
    def _check_final_overshoot(origins,lengths,min_length,direction_vector,box_size):  
        """
        Checks if there are any subsightline lengths that are too small (e.g. near corners of boxes).
        """      

        # Merge tiny lengths into the final segment
        final_lengths = []
        final_origins = []
        carry = 0.0
        for i,l in enumerate(lengths):
            if l >= min_length:
                final_lengths.append(l)
                final_origins.append(origins[i])
            else:
                carry += l
        final_lengths[-1] += carry  # add all tiny leftovers to the last segment

        # -- Generate ray and find where it is outside the box -- #
        final_ray = np.linspace(origins[-1],origins[-1]+direction_vector*final_lengths[-1],1000) 
        ray_out_box = final_ray[~((final_ray[:,0]<=box_size) & (final_ray[:,0]>=0) & 
                                (final_ray[:,1]<=box_size) & (final_ray[:,1]>=0) & 
                                (final_ray[:,2]<=box_size) & (final_ray[:,2]>=0))]

        # -- If outside, take where it is first outside box, calculate length remaining, and return new origin with box length subtracted -- #
        if len(ray_out_box) > 0:
            new_origin = ray_out_box[0]
            correct_index = np.argmax(abs(new_origin-box_size/2))
            if new_origin[correct_index] >box_size/2:
                new_origin[correct_index] = box_size
                outside_length = np.linalg.norm(ray_out_box[-1]-new_origin)
                new_origin[correct_index] = 0
            else:
                new_origin[correct_index] = 0
                outside_length = np.linalg.norm(ray_out_box[-1]-new_origin)
                new_origin[correct_index] = box_size

            if outside_length >= min_length:
                final_lengths.append(outside_length)
                final_origins.append(new_origin)


        return final_origins,final_lengths

    @staticmethod
    def _iterate_pacman_paths(origin,length,direction_vector,box_size,min_length=20):
        """
        Collects the origins and lengths of each sub sightline in this sightline.
        """

        current_origin = origin
        current_length = length

        origins = []
        lengths = []
        while current_origin is not None:
            origins.append(current_origin)
            lengths.append(current_length)
            current_origin,current_length = Sightline._find_exterior_rays(current_origin,current_length,direction_vector,box_size)        
        lengths.append(0)
        lengths = -1*np.diff(lengths)

        if len(lengths[lengths<min_length]) > 0:
            origins,lengths = Sightline._check_final_overshoot(origins,lengths,min_length,direction_vector,box_size)

        return origins,lengths

    def _discrete_snapshots(self,cosmo,redshifts,box_size,verbose):
        """
        Calculates origins and lengths of sightlines required to traverse different redshifts.
        """            

        # -- Finds the mid points of snapshot redshifts and distances -- #
        mid_redshifts = (redshifts[:-1]+redshifts[1:])/2   # z_i + z_i+1 / 2 
        mid_redshifts = mid_redshifts[:len(mid_redshifts[mid_redshifts<self.target_redshift])+1]

        mid_distances = cosmo.comoving_distance(mid_redshifts).value*1000

        snaps_required = len(mid_distances)

        # -- Generate list of distances traversed in each snapshots -- #
        if snaps_required > 1:
            discrete_distances = np.diff(mid_distances)
            discrete_distances = np.insert(discrete_distances,0,mid_distances[0])
            discrete_distances[-1] = self.length-mid_distances[-2]
        else:
            discrete_distances = [self.length]

        # -- Generate list of origin points for each snapshot used -- #
        discrete_origins = []
        for d in mid_distances[:-1]:
            discrete_origins.append(self.origin + self.direction_vector*d)
        discrete_origins = np.insert(discrete_origins,0,self.origin).reshape(snaps_required,3)

        # -- If any origins are outside the box, put them in the box -- #
        for i,origin in enumerate(discrete_origins):
            for j in range(3):
                if not (0 <= origin[j] <= box_size):
                    discrete_origins[i,j] -= box_size * (origin[j] // box_size)

        if verbose:
            print(f'Snapshots Needed: {snaps_required}',flush=True)

        return discrete_origins,discrete_distances,redshifts[:snaps_required]    
        
    def partition(self,sim,verbose=True):
        """
        Subdivides Sightline into the number of sightlines needed based on redshift traversal and pacman.
        """
        # -- Convert redshift to length or vice versa -- #
        self.sightline_distance(sim.cosmo)

        # -- Calculate origins and lengths for each snapshot -- #
        snapshot_origins,snapshot_distances,redshifts = self._discrete_snapshots(sim.cosmo,sim.redshifts,sim.box_size,verbose)

        # -- Allocate -- #
        self.sub_Origins = []
        self.sub_Lengths = []
        self.sub_BoxRedshifts = []
        self.sub_Snapshots = []

        # -- Partition each snapshot sightline -- #
        count = 0
        for i in range(len(snapshot_origins)):
            sub_origins,sub_lengths = self._iterate_pacman_paths(snapshot_origins[i],snapshot_distances[i],self.direction_vector,sim.box_size)
            for j in range(len(sub_origins)):

                self.sub_Origins.append(sub_origins[j])
                self.sub_Lengths.append(sub_lengths[j])
                self.sub_BoxRedshifts.append(redshifts[i])
                self.sub_Snapshots.append(i)

                count += 1

        self.num_sub_sightlines = count  

        self.sub_PointsIdx = [[] for i in range(self.num_sub_sightlines)]
        self.sub_Halos = [[] for i in range(self.num_sub_sightlines)]
        self.sub_Compute = [[] for i in range(self.num_sub_sightlines)]
        self.sub_Density = [[] for i in range(self.num_sub_sightlines)] 
        self.sub_Grid = [[] for i in range(self.num_sub_sightlines)]
        self.sub_Cells = [[] for i in range(self.num_sub_sightlines)]
        self.sub_HaloAssignment = [[] for i in range(self.num_sub_sightlines)]
        self.sub_CellConditions = [[] for i in range(self.num_sub_sightlines)]

        self.sub_Origins = np.array(self.sub_Origins).astype(np.float32)
        self.sub_Lengths = np.array(self.sub_Lengths).astype(np.float32)
        self.sub_BoxRedshifts = np.array(self.sub_BoxRedshifts)
        self.sub_Snapshots = np.array(self.sub_Snapshots).astype(int)

        return self



    # ------------- Altering/saving/extending/reducing sightline ------------- #


    def get_subsightline(self,idx,with_values=False):
        """
        Returns Sightline object for sub sightline at given idx.
        """

        subsightline = Sightline(length=self.sub_Lengths[idx],
                         direction_vector=self.direction_vector,
                         origin=self.sub_Origins[idx],
                         parent_sightline=False)
        
        if with_values:
            subsightline.sub_Grid = self.sub_Grid[idx]
            subsightline.sub_Compute = self.sub_Compute[idx]
            subsightline.sub_Density = self.sub_Density[idx]
            subsightline.sub_CellConditions = self.sub_CellConditions[idx]
            subsightline.sub_Halos = self.sub_Halos[idx]
            subsightline.sub_HaloAssignment = self.sub_HaloAssignment[idx]
            subsightline.sub_BoxRedshifts = self.sub_BoxRedshifts[idx]
            subsightline.sub_Snapshots = self.sub_Snapshots[idx]

        return subsightline
    
    def save(self,save_path,return_file=False):
        """
        Save sightline out to where it is so far.
        """

        # -- Make save path if needed -- #
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # -- Generate save file name -- #
        snapshot_reached = self.sub_Snapshots[self.subsightline_reached()-1]
        save_name = f'sightline_z{self.target_redshift:.3f}_snap{snapshot_reached}_{self.sightline_idx}'

        # -- Save to pkl file -- #
        with open(f'{save_path}/{save_name}.pkl','wb') as f:
            pickle.dump(self,f)

        # # -- Remove all previous sightlines -- #
        # all_saves = glob(f'{save_path}/*_{self.sightline_idx}.pkl')
        # for file in all_saves:
        #     if save_name not in file:
        #         os.system(f'rm {file}')
        
        if return_file:
            return f'{save_path}/{save_name}.pkl'

    def extend(self,sim,redshift):
        """
        Extends the number of subsightlines if self.target_redshift is changed, whilst retaining current information.
        """

        if redshift > self.target_redshift:
            sl_copy = deepcopy(self)
            sl_copy.target_redshift = redshift
            sl_copy.partition(sim,verbose=False)

            # n = self.num_sub_sightlines - 1
            n = self.subsightline_reached() - 1

            if n > 0:
                for i in range(n):
                    sl_copy.sub_PointsIdx[i] = self.sub_PointsIdx[i]
                    sl_copy.sub_Grid[i]      = self.sub_Grid[i]
                    sl_copy.sub_Cells[i]     = self.sub_Cells[i]
                    sl_copy.sub_Density[i]   = self.sub_Density[i]
                    sl_copy.sub_Compute[i]   = self.sub_Compute[i]
                    sl_copy.sub_Halos[i]     = self.sub_Halos[i]
                    # if self.sub_HaloAssignment is not None:     # temporary
                    sl_copy.sub_HaloAssignment[i]     = self.sub_HaloAssignment[i]
                    sl_copy.sub_CellConditions[i]     = self.sub_CellConditions[i]

            # Copy data back into self (mutate in place)
            self.num_sub_sightlines = sl_copy.num_sub_sightlines

            self.sub_PointsIdx          = sl_copy.sub_PointsIdx
            self.sub_Grid               = sl_copy.sub_Grid
            self.sub_Cells              = sl_copy.sub_Cells
            self.sub_Density            = sl_copy.sub_Density
            self.sub_Compute            = sl_copy.sub_Compute
            self.sub_Halos              = sl_copy.sub_Halos
            self.sub_HaloAssignment     = sl_copy.sub_HaloAssignment
            self.sub_CellConditions     = sl_copy.sub_CellConditions
            
            self.sub_Lengths = sl_copy.sub_Lengths
            self.sub_BoxRedshifts = sl_copy.sub_BoxRedshifts
            self.sub_Origins = sl_copy.sub_Origins
            self.sub_Snapshots = sl_copy.sub_Snapshots
            self.target_redshift = sl_copy.target_redshift
            self.length = sl_copy.length

    def _save_pointsidx(self,save_path):

        # -- Make save path if needed -- #
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_name = f'sightline_{self.sightline_idx}_PointsIdx.npy'

        # Load existing or initialise
        if os.path.exists(f'{save_path}/{save_name}'):
            saved = np.load(f'{save_path}/{save_name}', allow_pickle=True).tolist()
            if len(saved) < self.num_sub_sightlines:
                saved += [[] for _ in range(self.num_sub_sightlines - len(saved))]
        else:
            saved = [[] for _ in range(self.num_sub_sightlines)]

        # Update saved with any newly filled indices
        for i, pidx in enumerate(self.sub_PointsIdx):
            if len(pidx) > 0 and pidx[0] != 'Removed':
                saved[i] = pidx

        np.save(f'{save_path}/{save_name}', np.array(saved, dtype=object), allow_pickle=True)
    
    def reduce(self,grid_resolution,cgm_buffer,inplace=True,save_points_path=None):
        """
        Reduce resolution of IGM component and remmove PointsIdx. Only do after halo_assigment is complete.
        """

        if save_points_path is not None:
            self._save_pointsidx(save_points_path)

        new_Grid, new_Density, new_Compute, new_HaloAssignment, new_CellConditions,new_Cells,new_PointsIdx = [[[] for _ in range(self.num_sub_sightlines)] for _ in range(7)]
        arrs = [new_Grid, new_Density, new_Compute, new_HaloAssignment, new_CellConditions, new_Cells]

        def append_seg(i, s, e):
            arrs[0][i] = np.concatenate([arrs[0][i], self.sub_Grid[i][s:e]])
            arrs[1][i] = np.concatenate([arrs[1][i], self.sub_Density[i][s:e]])
            arrs[2][i] = np.concatenate([arrs[2][i], self.sub_Compute[i][s:e]])
            arrs[3][i] = np.concatenate([arrs[3][i], self.sub_HaloAssignment[i][s:e]])
            arrs[4][i] = np.concatenate([arrs[4][i], self.sub_CellConditions[i][s:e]])
            arrs[5][i] = np.concatenate([arrs[5][i], self.sub_Cells[i][s:e]])

        def append_igm(i, grid, density, compute):
            arrs[0][i] = np.concatenate([arrs[0][i], grid])
            arrs[1][i] = np.concatenate([arrs[1][i], density])
            arrs[2][i] = np.concatenate([arrs[2][i], compute])
            arrs[3][i] = np.concatenate([arrs[3][i], np.ones_like(grid)*-1])
            arrs[4][i] = np.concatenate([arrs[4][i], np.zeros_like(grid)])
            arrs[5][i] = np.concatenate([arrs[5][i], np.ones_like(grid)*-1])

        n_subsightlines = self.subsightline_reached(grid=True,halos=True)
        for i in range(n_subsightlines):

            new_PointsIdx[i] = ['Removed']

            median_length = np.nanmedian(self.sub_Grid[i])
            downsample_factor = int(grid_resolution // median_length)
            if downsample_factor <= 1:
                for k, src in enumerate([self.sub_Grid[i], self.sub_Density[i], self.sub_Compute[i],
                                        self.sub_HaloAssignment[i], self.sub_CellConditions[i], self.sub_Cells[i]]):
                    arrs[k][i] = src
                continue

            is_igm = (self.sub_CellConditions[i] == 0)
            switches = np.diff(is_igm.astype(int), prepend=0, append=0)
            igm_starts = np.where(switches ==  1)[0]
            igm_stops  = np.where(switches == -1)[0]

            # -- Initialise arrays -- #
            if len(igm_starts) == 0:
                for k, src in enumerate([self.sub_Grid[i], self.sub_Density[i], self.sub_Compute[i], self.sub_HaloAssignment[i], self.sub_CellConditions[i], self.sub_Cells[i]]):
                    arrs[k][i] = src
                continue
            for k in range(6):
                arrs[k][i] = np.empty(0)

            # -- Prepend any CGM before the first IGM segment -- #
            append_seg(i, 0, igm_starts[0])  

            # -- Go through each IGM segment and downsample -- #
            for j, (start, stop) in enumerate(zip(igm_starts, igm_stops)):
                has_left_cgm  = start > 0
                has_right_cgm = stop < len(self.sub_Grid[i])

                buf_start = min(start + cgm_buffer, stop) if has_left_cgm  else start
                buf_stop  = max(stop - cgm_buffer, start) if has_right_cgm else stop

                append_seg(i, start, buf_start)  # left buffer (full res)

                if buf_stop > buf_start:          # downsampled core
                    idx = np.arange(0, buf_stop - buf_start, downsample_factor)
                    grid = np.add.reduceat(self.sub_Grid[i][buf_start:buf_stop], idx)
                    compute = np.add.reduceat(self.sub_Compute[i][buf_start:buf_stop], idx)

                    # counts = np.diff(np.append(idx, buf_stop - buf_start))
                    # density = np.add.reduceat(self.sub_Density[i][buf_start:buf_stop], idx) / counts

                    dl               = self.sub_Grid[i][buf_start:buf_stop]
                    weighted_density = np.add.reduceat(self.sub_Density[i][buf_start:buf_stop] * dl, idx)
                    total_dl         = np.add.reduceat(dl, idx)
                    density          = weighted_density / total_dl


                    append_igm(i, grid, density, compute)

                append_seg(i, buf_stop, stop)    # right buffer (full res)

                if j + 1 < len(igm_starts):  # CGM between segments
                    append_seg(i, stop, igm_starts[j+1])  

            # -- Append any CGM after the final IGM segment -- #
            append_seg(i, igm_stops[-1], len(self.sub_Grid[i]))  

        if inplace:
            self.sub_Grid[:n_subsightlines]           = new_Grid[:n_subsightlines]
            self.sub_Density[:n_subsightlines]        = new_Density[:n_subsightlines]
            self.sub_Compute[:n_subsightlines]        = new_Compute[:n_subsightlines]
            self.sub_HaloAssignment[:n_subsightlines] = new_HaloAssignment[:n_subsightlines]
            self.sub_CellConditions[:n_subsightlines] = new_CellConditions[:n_subsightlines]
            self.sub_Cells[:n_subsightlines]          = new_Cells[:n_subsightlines]
            self.sub_PointsIdx[:n_subsightlines]      = new_PointsIdx[:n_subsightlines]
            
        else:
            sl = deepcopy(self)
            sl.sub_Grid[:n_subsightlines]           = new_Grid[:n_subsightlines]
            sl.sub_Density[:n_subsightlines]        = new_Density[:n_subsightlines]
            sl.sub_Compute[:n_subsightlines]        = new_Compute[:n_subsightlines]
            sl.sub_HaloAssignment[:n_subsightlines] = new_HaloAssignment[:n_subsightlines]
            sl.sub_CellConditions[:n_subsightlines] = new_CellConditions[:n_subsightlines]
            sl.sub_Cells[:n_subsightlines]          = new_Cells[:n_subsightlines]
            sl.sub_PointsIdx[:n_subsightlines]      = new_PointsIdx[:n_subsightlines]
            return sl

    # ------------- Sightline information readout / plotting ------------- #

    def readout_subsightlines(self,number=None):
        """
        Prints out sub sightline information.
        """

        if self.num_sub_sightlines is not None:
            number = min(v for v in [self.num_sub_sightlines, number] if v is not None)

            for i in range(number):

                points_idx = self.sub_PointsIdx[i]
                if len(points_idx) > 4:
                    display_points = list(points_idx[:4]) + ["..."]
                else:
                    display_points = list(points_idx)

                if len(self.sub_Halos[i]) > 4:
                    display_radii = [h['Radius'] for h in self.sub_Halos[i][:4]] + ["..."]
                else:
                    display_radii = [h['Radius'] if h is not None else None for h in self.sub_Halos[i]]

                msg = f"Sub sightline #{i}: Origin = {self.sub_Origins[i]}\n\
                  Length = {self.sub_Lengths[i]}\n\
                  Box Redshift = {self.sub_BoxRedshifts[i]}\n\
                  Snapshot = {self.sub_Snapshots[i]}\n\
                  PointsIdx = {display_points} (length = {len(points_idx)})\n\
                  Halos: Radii:{display_radii} (length = {sum(1 for h in self.sub_Halos[i] if h is not None)})"

                if len(self.sub_Compute[i]) > 0:
                    msg += f"\n\
                  Grid = {list(self.sub_Grid[i][:4]) + ['...']}\n\
                  Cells = {list(self.sub_Cells[i][:4]) + ['...']}\n\
                  Density = {list(self.sub_Density[i][:4]) + ['...']}\n\
                  Compute = {list(self.sub_Compute[i][:4]) + ['...']})"
                    
                if self.sub_HaloAssignment is not None:
                    msg += f"\n\
                  HaloAssignment = {list(self.sub_HaloAssignment[i][:4]) + ['...']}\n\
                  CellConditions = {list(self.sub_CellConditions[i][:4]) + ['...']})"
                  

                print(msg)
                print('\n')

        else:
            raise ValueError('No sub sightlines! Run .partition()!')
        
    def plot_subsightline(self,idx,points=None,halos=True,compute=True):
        """
        Plot the halos / coordinates in the transformed basis of a given sub sightline. Also plots the "compute".
        """

        if (points is None) & (halos is None):
            e = "Provide either points or halos! What's the point otherwise??"
            raise ValueError(e)
        
        else:

            zlims = [0,self.sub_Lengths[idx]]

            fig, ax = plt.subplots(ncols=3,figsize=(12,8))
            if compute:
                if len(self.sub_Compute[0]) > 0:       
                    plt.close()
                    fig, ax = plt.subplots(ncols=4,figsize=(12,8))

                    positions = np.concatenate([[0], np.cumsum(self.sub_Grid[idx])])
                    y = np.repeat(positions, 2)[1:-1]   # duplicate edges, drop first and last to match density
                    x = np.repeat(self.sub_Compute[idx], 2) 
                    ax[3].plot(x,y)
                    ax[3].set_ylim(zlims[0],zlims[1])
                    ax[3].set_xlabel('Computed Value')
                    ax[3].set_ylabel('z [ckpc]')

            subSL = self.get_subsightline(idx)

            ax[0].axvline(0,alpha=0.5,c='k')
            ax[0].axhline(0,alpha=0.5,c='k')
            ax[0].set_aspect('equal')
            ax[0].set_xlabel('x [ckpc]')
            ax[0].set_ylabel('y [ckpc]')

            ax[1].axvline(0,alpha=0.5,c='k')
            ax[1].set_aspect('equal')
            ax[1].set_xlabel('x [ckpc]')
            ax[1].set_ylabel('z [ckpc]')
            ax[1].set_ylim(zlims[0],zlims[1])

            ax[2].axvline(0,alpha=0.5,c='k')
            ax[2].set_aspect('equal')
            ax[2].set_xlabel('y [ckpc]')
            ax[2].set_ylabel('z [ckpc]')
            ax[2].set_ylim(zlims[0],zlims[1])
        
            if points is not None:
                in_points = points[self.sub_PointsIdx[idx]]
                transformed_points = Transform_Points(subSL,in_points)
                ax[0].scatter(transformed_points[:,0],transformed_points[:,1],s=1)
                ax[1].scatter(transformed_points[:,0],transformed_points[:,2],s=1)
                ax[2].scatter(transformed_points[:,1],transformed_points[:,2],s=1)

            if halos:

                if any(h is not None for h in self.sub_Halos[idx]):

                    coms = np.array([h['Pos'] for h in self.sub_Halos[idx]])
                    coms = Transform_Points(subSL,coms)

                    max_r = max(max(h['Radius'] for h in self.sub_Halos[idx]),500)
                    max_xy = np.max(np.abs(coms[:, :2])) + max_r
                
                    for i in range(len(self.sub_Halos[idx])):
                        x,y,z = coms[i]

                        patch = Circle((x,y),radius=self.sub_Halos[idx][i]['Radius'],facecolor='none',edgecolor='r',alpha=0.5)
                        ax[0].add_patch(patch)

                        patch = Circle((x,z),radius=self.sub_Halos[idx][i]['Radius'],facecolor='none',edgecolor='r',alpha=0.5)
                        ax[1].add_patch(patch)

                        patch = Circle((y,z),radius=self.sub_Halos[idx][i]['Radius'],facecolor='none',edgecolor='r',alpha=0.5)
                        ax[2].add_patch(patch)
                    
                    ax[0].set_xlim(-max_xy, max_xy)
                    ax[0].set_ylim(-max_xy, max_xy)
                    ax[1].set_xlim(-max_xy, max_xy)
                    ax[2].set_xlim(-max_xy, max_xy)

                    
    def plot_compute(self,idx=None,data='compute',with_model=False,return_data=False,logspace=True,mode='normal'):
        """
        Combines all computed subsightlines into one non-cumulutive profile, coloured by subsightline.
        """

        if data.lower() not in ('compute', 'density'):
            raise ValueError('data must be "compute" or "density"!')

        attr = f"sub_{data.capitalize()}"

        lengths  = [self.sub_Grid[idx]] if idx is not None else self.sub_Grid[:self.subsightline_reached()]
        computed = [getattr(self, attr)[idx]] if idx is not None else getattr(self, attr)[:self.subsightline_reached()]
        cmap = [_Get_Colours(self.num_sub_sightlines)[idx]] if idx is not None else _Get_Colours(self.subsightline_reached())


        if with_model:
            max_idx = self.subsightline_reached(modelled=True)
            if max_idx > 0 and self.modelled is not None:
                model_lengths = [self.modelled.sub_Grid[idx]] if idx is not None else self.modelled.sub_Grid
                model_computed = [getattr(self.modelled, attr)[idx]] if idx is not None else getattr(self.modelled, attr)
                if data == 'compute' and mode != 'cumulutive':
                    print('Model is on resolution with different grid! Values will look weird!')
                elif data == 'density' and mode == 'cumulutive':
                    print('Model is on resolution with different grid! Values will look weird!')
            else:
                with_model = False
                
        plt.figure(figsize=(8,4))
        plt.xlabel("Distance along ray")
        
        if data == 'compute':
            if mode == 'cumulutive':
                plt.ylabel('Cumulutive DM')
            else:
                plt.ylabel('DM')
        else:
            if mode == 'cumulutive':
                plt.ylabel('Cumulutive Density')
            else:
                plt.ylabel('Density')

        if logspace:
            plt.yscale('log')

        xs = []
        ys = []
        cumulated_length = 0
        model_cumulated_value = 0
        cumulated_value = 0
        for i in range(len(lengths)):
            x = np.cumsum(lengths[i]) + cumulated_length
            if mode != 'cumulutive':
                y = computed[i]
            else:
                y = np.cumsum(computed[i])

            plt.step(x, y+cumulated_value, color=cmap[i], where='post')

            if with_model:
                model_x = np.cumsum(model_lengths[i]) + cumulated_length
                if mode != 'cumulutive':
                    model_y = model_computed[i]
                else:
                    model_y = np.cumsum(model_computed[i])
                    
                plt.step(model_x, model_y+model_cumulated_value, color='k', alpha=0.5, linestyle='--')

            cumulated_length += np.nansum(lengths[i])
            xs.extend(x)
            ys.extend(y)
            if mode == 'cumulutive':
                cumulated_value += y[-1]
                model_cumulated_value += model_y[-1]

            plt.axvline(x[-1],alpha=0.3,c=cmap[i],linestyle=':')

        plt.xlim(0,cumulated_length)

        if return_data:
            return np.array(xs),np.array(ys)
        


    # ------------- Computation / environment partitioning ------------- #

    def assign_to_halos(self,method='sphere',particle_ids=None,snapshot=None,sim=None):
        """
        Assign grid cells to halos. Either through spherical r200c check, or through halo membership.
        """

        self.halos_traversed = {}

        num_sub_sightlines = self.subsightline_reached(halos=True)

        for i in range(num_sub_sightlines):

            self.sub_HaloAssignment[i] = np.full(self.sub_Grid[i].shape[0],-1)
            self.sub_CellConditions[i] = np.full(self.sub_Grid[i].shape[0],0)

            if self.sub_Halos[i] == [None]:
                continue
            

            if method == 'sphere':

                subSL = self.get_subsightline(i)
                
                z_edges = np.concatenate(([0.0], np.cumsum(self.sub_Grid[i]))) # z-midpoints of each segment
                z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])  

                # Extract halo data
                halo_positions = np.array([Transform_Points(subSL,halo['Pos']) for halo in self.sub_Halos[i] if halo is not None])  
                radii = np.array([halo['Radius'] for halo in self.sub_Halos[i]])
                ids = np.array([halo['ID'] for halo in self.sub_Halos[i]])

                # Distance from (0,0,z_mid) to halo COM
                dx2 = halo_positions[:, 0][:, None]**2
                dy2 = halo_positions[:, 1][:, None]**2
                dz2 = (halo_positions[:, 2][:, None] - z_mid[None, :])**2
                dist2 = dx2 + dy2 + dz2

                # Find grid points inside halos
                inside = dist2 <= radii[:, None]**2     # (Nhalo, Nseg)

                for j in range(inside.shape[0]):
                    self.sub_HaloAssignment[i] += inside[j,:].astype(int) * (ids[j]+1)

                self.sub_CellConditions[i][np.where(self.sub_HaloAssignment[i]>-1)[0]] = 1

            elif method == 'particles':

                if self.sub_Snapshots[i] != snapshot:
                    continue
                
                cell_ids = particle_ids[self.sub_Cells[i]]

                for halo in self.sub_Halos[i]:

                    halo_info = sim.load_halo(sim._get_snap_num(snapshot),halo['ID'])  # Load in info for this halo
                    
                    inside_array = np.isin(cell_ids, halo_info['ParticleIDs'])
                    self.sub_HaloAssignment[i][inside_array] = halo['ID']
                    self.sub_CellConditions[i][inside_array] = 1

    def subsightline_reached(self,grid=True,halos=False,assigned=False,observed=False,modelled=False):

        checks = []                 
        if grid:
            #checks.append(next((i for i, h in enumerate(self.sub_Grid)  if len(h) == 0), self.num_sub_sightlines))
            checks.append(next((i for i, h in enumerate(self.sub_Grid)  if len(h) == 0), len(self.sub_Grid)))
        if halos:
            checks.append(next((i for i, h in enumerate(self.sub_Halos) if h == []), self.num_sub_sightlines))
        if assigned:
            checks.append(next((i for i, h in enumerate(self.sub_CellConditions) if len(h) == 0), self.num_sub_sightlines))
        if observed:
            if self.sub_Observed is not None:
                checks.append(next((i for i, h in enumerate(self.sub_Observed) if not h), self.num_sub_sightlines))
                # checks.append(next((i for i, h in enumerate(self.observed.sub_Halos) if h == []), self.num_sub_sightlines))
            else:
                checks.append(0)
        if modelled:
            if self.modelled is not None:
                checks.append(next((i for i, h in enumerate(self.modelled.sub_Grid)  if len(h) == 0), self.num_sub_sightlines))
            else:
                checks.append(0)
        
        num_sub_sightlines = min(checks) 

        return num_sub_sightlines

    def redshift_reached(self,cosmo,environment='total',observed=False,modelled=False):
        """
        Determine maximum redshift reached in computation.
        """

        if environment == 'total':
            halos = False
        else:
            halos = True

        num_sub_sightlines = self.subsightline_reached(grid=True,halos=halos,observed=observed,modelled=modelled)
        valid = [sl for sl in self.sub_Grid[:num_sub_sightlines] if len(sl) > 0]
        full_grid = np.nansum(np.concatenate(valid)) if valid else 0.0

        return z_at_value(cosmo.comoving_distance, full_grid*u.kpc).value if full_grid > 0. else 0.

    def extract_compute(self, cosmo, redshift=None, environment='total',modelled=False):
        """
        Extract cumulutive computed results as a function of redshift and environment.
        """

        if (environment != 'total') & (self.sub_CellConditions is None):
            e = 'Cell partitioning to igm / cgm not complete yet! Call "self.assign_to_halos()"'
            raise ValueError(e)

        condition_map = {'total': [0, 1], 'igm': [0], 'cgm': [1]}
        good_conditions = condition_map[environment]

        if environment == 'total':
            halos = False
        else:
            halos = True

        num_sub_sightlines = self.subsightline_reached(grid=True,halos=halos,modelled=modelled)
        sls = slice(None, num_sub_sightlines)

        grid = getattr(self.modelled, 'sub_Grid') if modelled else getattr(self, 'sub_Grid')
        compute = getattr(self.modelled, 'sub_Compute') if modelled else getattr(self, 'sub_Compute')
        conditions = getattr(self.modelled, 'sub_CellConditions') if modelled else getattr(self, 'sub_CellConditions')

        full_grid  = np.cumsum(np.concatenate(grid[sls]))
        full_compute    = np.concatenate(compute[sls])
        cell_conditions = np.concatenate(conditions[sls]) if environment != 'total' else np.zeros(len(full_compute))
        
        if redshift is None:
            redshift = self.redshift_reached(cosmo,environment,modelled=modelled)

        z = np.atleast_1d(redshift)
        lengths = cosmo.comoving_distance(z).value * 1000.0

        values = []
        for L in lengths:
            i = min(np.searchsorted(full_grid, L), len(full_grid) - 1)
            values.append(np.nansum(full_compute[:i][np.isin(cell_conditions[:i],good_conditions)]))

        return values[0] if np.isscalar(redshift) else np.array(values)
    

    def halo_info(self,modelled=False,with_compute=False):

        halos_traversed = {}

        num_sub_sightlines = self.subsightline_reached(grid=False,halos=True,modelled=modelled,assigned=with_compute)

        source = self.modelled if modelled else self
        
        for i in range(num_sub_sightlines):
            if source.sub_Halos[i] != [None]:
                for halo in source.sub_Halos[i]:
                    halos_traversed[halo['ID']] = halo
                    halos_traversed[halo['ID']]['SubSightline'] = i
                    if with_compute:
                        halos_traversed[halo['ID']]['Compute'] = np.nansum(source.sub_Compute[i][source.sub_HaloAssignment[i]==halo['ID']])

                
        return halos_traversed


    # ------------- Observing / Modelling Sightline ------------- #

    def _initialise_modelled(self,inference):

        mod = Sightline(origin=self.origin,
                    direction_vector=self.direction_vector,
                    length=self.length,
                    parent_sightline=False)
    
        mod.num_sub_sightlines = self.num_sub_sightlines
        mod.sub_Origins = self.sub_Origins
        mod.sub_Lengths = self.sub_Lengths
        mod.sub_BoxRedshifts = self.sub_BoxRedshifts
        mod.sub_Snapshots = self.sub_Snapshots
        mod.sub_Halos = self.sub_Halos 

        # These get populated by the inference pipeline
        mod.sub_Density = [[] for _ in range(self.num_sub_sightlines)]
        mod.sub_Compute = [[] for _ in range(self.num_sub_sightlines)]        # inferred DM
        mod.sub_Grid = [[] for _ in range(self.num_sub_sightlines)]           # can mirror original
        mod.sub_CellConditions = [[] for _ in range(self.num_sub_sightlines)]
        mod.sub_HaloAssignment = [[] for _ in range(self.num_sub_sightlines)]

        self.modelled = mod
        self.modelled.model_params = inference.model_params

    def observe_halos(self, galaxyfinder, grid_path, filters=['lsst_g','lsst_r','lsst_i','lsst_z']):

        limiting_mags = np.array([LIMITING_MAGS[key] for key in filters])

        sub_ObservedHalos = [[] for _ in range(self.num_sub_sightlines)]

        self.sub_Observed = [False for _ in range(self.num_sub_sightlines)]

        num_sub_sightlines = self.subsightline_reached(grid=False,halos=True)

        for i in range(num_sub_sightlines):
            if self.sub_Snapshots[i] == galaxyfinder.snapshot:
                if self.sub_Halos[i] != [None]:
                    for halo in self.sub_Halos[i]:

                        prelength = np.nansum(self.sub_Lengths[:i])
                        subsl = self.get_subsightline(i)
                        halo_dist = Transform_Points(subsl, halo['Pos'])[2]
                        redshift = z_at_value(galaxyfinder.sim.cosmo.comoving_distance, (prelength + halo_dist) * u.kpc)

                        halo_info = deepcopy(halo)
                        halo_info['Redshift'] = redshift.value

                        if halo['NumStars'] >= galaxyfinder.nstars_limit:

                            observed_halo = galaxyfinder.process_halo(halo['ID'], subsl, redshift, grid_path, filters, apply_dust=True, plot=False, verbose=False)

                            n = len(observed_halo['GalaxyStellarMasses'])

                            visible_per_filter = {
                                f: [observed_halo['GalaxyApparentMags'][f][j] <= mag for j in range(n)]
                                for f, mag in zip(filters, limiting_mags)
                            }

                            halo_info['ObservedGalaxies'] = []
                            for j in range(n):
                                in_all = all(visible_per_filter[f][j] for f in filters)
                                in_any = any(visible_per_filter[f][j] for f in filters)

                                halo_info['ObservedGalaxies'].append({
                                    'StellarMass':      observed_halo['GalaxyStellarMasses'][j],
                                    'ApparentMags':     {f: observed_halo['GalaxyApparentMags'][f][j] for f in filters},
                                    'AbsoluteMags':     {f: observed_halo['GalaxyAbsoluteMags'][f][j] for f in filters},
                                    'MassLightRatio':   observed_halo['GalaxyMassLightRatio'][j],
                                    'VisiblePerFilter': {f: visible_per_filter[f][j] for f in filters},
                                    'Visible':          1 if in_all else (0 if in_any else -1),
                                })

                        else:                            
                            halo_info['ObservedGalaxies'] = []

                        sub_ObservedHalos[i].append(halo_info)

                else:
                    sub_ObservedHalos[i] = [None]

            self.sub_Observed[i] = True

        self.sub_Halos = sub_ObservedHalos

        return self

    def infer_halos(self, inference, filters):

        limiting_mags = np.array([LIMITING_MAGS[key] for key in filters])

        num_sub_sightlines = self.subsightline_reached(grid=False,observed=True)

        for i in range(num_sub_sightlines):

            if self.sub_Halos[i] != [None]:
                for halo in self.sub_Halos[i]:
                    num_galaxies = len(halo['ObservedGalaxies'])

                    for j in range(num_galaxies):
                        
                        redshift = inference.infer_redshift(halo['Redshift'])
                        halo['ObservedGalaxies'][j]['Inferred_Redshift'] = redshift

                        if halo['ObservedGalaxies'][j]['Visible'] >= 0:
                            gal_abs_mags = inference.infer_galaxy_mags(halo['ObservedGalaxies'][j]['Inferred_Redshift'],
                                                                       halo['ObservedGalaxies'][j]['ApparentMags'],
                                                                       filters,limiting_mags)
                            gal_mass = inference.infer_galaxy_mass(gal_abs_mags)
                        else:
                            gal_abs_mags = None
                            gal_mass = None

                        halo['ObservedGalaxies'][j]['Inferred_StellarMass'] = gal_mass
                        halo['ObservedGalaxies'][j]['Inferred_AbsoluteMags'] = gal_abs_mags

                    inferred_gal_masses = [halo['ObservedGalaxies'][j]['Inferred_StellarMass'] for j in range(num_galaxies)]

                    halo['Inferred_TotalMass'] = inference.infer_halo_mass(inferred_gal_masses)
                    halo['Inferred_Radius'] = inference.infer_halo_size(halo['Inferred_TotalMass']) 

        self.halo_inference_params = inference.halo_inference_params

    def model_sightline(self, inference, filters=None,verbose=True):

        self._initialise_modelled(inference)

        if halo_params == 'inferred':
            num_sub_sightlines = self.subsightline_reached(grid=True,observed=True)
            if num_sub_sightlines == 0:
                if verbose:
                    print("Halos have not been observed! Switching 'halo_params' to 'truth'",flush=True)
                halo_params = 'truth'
            else:
                self.infer_halos(inference,filters)

        elif halo_params in ['truth',False]:
            num_sub_sightlines = self.subsightline_reached(grid=True,halos=True)  

        else:
            raise ValueError("halo_params must be 'inferred', 'truth', or set to False.")

        for i in range(num_sub_sightlines):
            subsightline = self.get_subsightline(i,with_values=True)
                    
            grid,density,compute,conditions,assign = inference.model_dm(subsightline,igm_background,halo_params,density_smooth_kernel)
            
            self.modelled.sub_Grid[i] = grid
            self.modelled.sub_Density[i] = density
            self.modelled.sub_Compute[i] = compute
            self.modelled.sub_CellConditions[i] = conditions
            self.modelled.sub_HaloAssignment[i] = assign
