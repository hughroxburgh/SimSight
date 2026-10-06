import numpy as np
from numba import njit,prange
from time import time as clock

# -- If finding points without a KDTree -- #

def _Points_Inside_Cylinder(points, origin, transformation_matrix, length, radius):
    """
    Find points inside a cylinder of a given radius.
    """

    inv_mat = np.linalg.inv(transformation_matrix)  # single inversion, done once
    newPoints = (points - origin) @ inv_mat.T       # batch transform all points
    inside_mask = (0 < newPoints[:, 0]) & (newPoints[:, 0] < length) & \
                  (np.sqrt(newPoints[:, 1]**2 + newPoints[:, 2]**2) < radius)
    return inside_mask

def _Gen_Cubic_Volume_Limits(circle1,circle2):
    """
    FOR CREATING CYLINDER - generates rectangular box limits around circular cylinder volume.
    """

    maxs1 = np.nanmax(circle1,axis=0)
    mins1 = np.nanmin(circle1,axis=0)

    maxs2 = np.nanmax(circle2,axis=0)
    mins2 = np.nanmin(circle2,axis=0)

    maxes = np.array([maxs1,maxs2])
    maxes = np.nanmax(maxes,axis=0)

    mins = np.array([mins1,mins2])
    mins = np.nanmin(mins,axis=0)

    xs = [mins[0],maxes[0]]
    ys = [mins[1],maxes[1]]
    zs = [mins[2],maxes[2]]

    return xs,ys,zs

def _Gen_Cross_Section(sightline, radius, resolution=1000):
    """
    Generates two circular cross-sections (bases of a cylinder) from a sightline definition.
    """
    theta = np.linspace(0, 2 * np.pi, resolution)
    # Use outer product to get all 3D positions of the circle in one go
    circle = (radius * np.cos(theta)[:, None] * sightline.basis_vectors[1] +
              radius * np.sin(theta)[:, None] * sightline.basis_vectors[2] +
              sightline.origin)
    circle2 = circle + sightline.length * sightline.basis_vectors[0]

    return circle, circle2

@njit(parallel=True)
def _Points_Subset(limits, points):
    xs, ys, zs = limits
    n = points.shape[0]
    subset = np.empty(n, dtype=np.bool_)

    for i in prange(n):
        x, y, z = points[i]
        subset[i] = (xs[0] < x < xs[1]) and (ys[0] < y < ys[1]) and (zs[0] < z < zs[1])     # mask for if in box

    return subset





# -- For finding points using a KDTree --# 

def _Points_Near_Ray_Tree(tree, 
                          ray_origin, ray_length, ray_direction, 
                          radii, coarse_radius,
                          giant_idx,giant_pts,giant_radii):
    """
    Find all points whose radius includes overlaps with a ray.
    """
    # -- Generate a ray, sampling enough times to ensure no point that lies near ray is outside a point+coarse_radius -- #
    nsteps = max(int(ray_length / coarse_radius * 3), 2)
    ray_points = np.linspace(ray_origin, ray_origin + ray_length * ray_direction, nsteps)
    
    # -- Find initial candidate idx based on a simple ball point query -- #
    candidates_idx = set()
    for point in ray_points:
        candidates_idx.update(tree.query_ball_point(point, r=coarse_radius))  # coarse search
    candidates_idx = np.array(list(candidates_idx), dtype=np.int64)

    # -- To make above ball point query easier, omit massive cells and now query them individually here -- #
    if len(giant_idx) > 0:
        p_vec = giant_pts - ray_origin
        t = np.clip(p_vec @ ray_direction, 0, ray_length)
        dist_sq = np.sum((p_vec - t[:, np.newaxis] * ray_direction)**2, axis=1)
        giant_hits = giant_idx[dist_sq <= giant_radii**2]
        candidates_idx = np.union1d(candidates_idx, giant_hits)

    if len(candidates_idx) == 0:
        return np.array([], dtype=np.int64)
    
    # -- Refine points and radii based on ball point query -- #
    points = tree.data[candidates_idx]
    radii = radii[candidates_idx]

    # -- Finally, refine fully based on individual radii of each point -- #
    p_vec = points - ray_origin  # vectors from ray start to points
    t = np.clip(p_vec @ ray_direction, 0, ray_length)    # projection along ray, clamped to segment
    dist_sq = np.sum((p_vec - t[:, np.newaxis] * ray_direction)**2, axis=1)  # distance to closest point on segment
    mask = dist_sq <= radii**2 # Step 3: Keep only points within radius

    return candidates_idx[mask]


# -- For finding points using a VoxelGrid --# 

def _Points_Near_Ray_Voxel(architecture, 
                           ray_origin, ray_length, ray_direction,
                            radii,voxel_size,
                            giant_idx, giant_pts, giant_radii):
    """
    Find all points whose radius overlaps with a ray, using a voxel grid.
    """

    order = architecture['order']          # point indices sorted by voxel (see Build_Voxel_Grid)
    vox_offsets = architecture['offsets']  # points in voxel k: order[vox_offsets[k]:vox_offsets[k+1]]
    coords = architecture['coords']        # (N,3) array of point coordinates
    grid_size = architecture['grid_size']  # number of voxels along each axis

    # -- Walk the ray, collecting candidate voxels -- #
    nsteps = max(int(ray_length / voxel_size * 1.5), 2)
    ray_points = np.linspace(ray_origin, ray_origin + ray_length * ray_direction, nsteps)
    ray_ijk = np.floor(ray_points / voxel_size).astype(np.int32)  # (nsteps, 3)

    # -- Compute flat keys for all 27 neighbours of every ray step at once -- #
    offsets = np.array([(di, dj, dk)
                     for di in (-1, 0, 1)
                     for dj in (-1, 0, 1)
                     for dk in (-1, 0, 1)], dtype=np.int32)
    nb_ijk    = (ray_ijk[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    flat_keys = (nb_ijk[:, 0].astype(np.int64) * grid_size * grid_size +
                 nb_ijk[:, 1].astype(np.int64) * grid_size +
                 nb_ijk[:, 2].astype(np.int64))

    # -- Collect candidate indices from occupied voxels -- #
    keys = np.unique(flat_keys)
    keys = keys[(keys >= 0) & (keys < len(vox_offsets) - 1)]
    keys = keys[vox_offsets[keys + 1] > vox_offsets[keys]]
    arrays = [order[vox_offsets[k]:vox_offsets[k + 1]] for k in keys]
    candidates_idx = np.unique(np.concatenate(arrays)) if arrays else np.array([], dtype=np.int64)

    # -- Add giants, tested individually -- #
    if len(giant_idx) > 0:
        p_vec    = giant_pts - ray_origin
        t        = np.clip(p_vec @ ray_direction, 0, ray_length)
        dist_sq  = np.sum((p_vec - t[:, np.newaxis] * ray_direction)**2, axis=1)
        giant_hits = giant_idx[dist_sq <= giant_radii**2]
        candidates_idx = np.union1d(candidates_idx, giant_hits)

    if len(candidates_idx) == 0:
        return np.array([], dtype=np.int64)

    # -- Refine using individual radii -- #
    pts     = coords[candidates_idx]
    r       = radii[candidates_idx]
    p_vec   = pts - ray_origin
    t       = np.clip(p_vec @ ray_direction, 0, ray_length)
    dist_sq = np.sum((p_vec - t[:, np.newaxis] * ray_direction)**2, axis=1)
    mask    = dist_sq <= r**2

    return candidates_idx[mask]



# -- Voxel grid -- #

def Build_Voxel_Grid(coords, voxel_size, box_size, exclude=None):
    """
    Sort points into a regular grid of voxels (flat key (i * grid_size + j) * grid_size + k). The indices of the
    points in voxel `key` are order[offsets[key]:offsets[key + 1]]. Points with exclude = True (e.g. the largest
    kernels, handled separately) are left out.
    """
    from ._utils import _Counting_Sort

    grid_size = int(np.ceil(box_size / voxel_size))
    if grid_size**3 >= 2**31:
        raise ValueError(f'voxel grid too fine ({grid_size}^3 voxels)')

    flat = np.zeros(len(coords), dtype=np.int32)
    for axis, mult in ((0, grid_size * grid_size), (1, grid_size), (2, 1)):
        ijk = (coords[:, axis] / voxel_size).astype(np.int32)
        np.clip(ijk, 0, grid_size - 1, out=ijk)
        ijk *= mult
        flat += ijk
        del ijk
    if exclude is not None:
        flat[exclude] = -1

    order, offsets = _Counting_Sort(flat, grid_size**3)
    del flat

    return {'order': order, 'offsets': offsets, 'grid_size': grid_size, 'voxel_size': float(voxel_size),
            'coords': coords}


# -- Gas within R200c of crossed halos -- #

X_H, MU_H, MU_E = 0.76, 1.3, 1.167     # as in _compute.Calc_Ray_DM (X_H) and _compute.Density_To_DM (MU_H, MU_E)


def Halo_Gas_Weights(data):
    """
    Per-particle weights for the halo gas sums: gas mass [1e10 Msun], and DM-equivalent gas mass (non-star-forming
    gas weighted by ionisation, as in Calc_Ray_DM) -- the mass the fully ionised Density_To_DM needs to reproduce
    the truth free-electron density.
    """
    w_gas = np.asarray(data['Masses'], dtype=np.float32)
    w_dm = (w_gas * (data['StarFormationRate'] == 0) * data['ElectronAbundance'] * (X_H * MU_H / MU_E)).astype(np.float32)
    return w_gas, w_dm


@njit(parallel=True, cache=True, fastmath=True)
def _Sum_Gas_In_Spheres(centres, radii, coords, w_gas, w_dm, order, offsets, grid_size, voxel_size, box):
    """Sum w_gas / w_dm over the points within radii of centres (periodic), visiting only the voxels each sphere touches."""
    n = radii.shape[0]
    gas = np.zeros(n)
    dm = np.zeros(n)
    count = np.zeros(n, dtype=np.int64)
    for h in prange(n):
        R = radii[h]
        if R <= 0.0:
            continue
        R2 = R * R

        # -- voxel index ranges covering the sphere along each axis: the in-box part, plus wrapped parts -- #
        c = np.empty(3)
        ranges = np.empty((3, 3, 2), dtype=np.int64)
        n_ranges = np.zeros(3, dtype=np.int64)
        for a in range(3):
            ca = centres[h, a] % box
            c[a] = ca
            lo, hi = ca - R, ca + R
            m = 0
            ranges[a, m, 0] = min(int(max(lo, 0.0) / voxel_size), grid_size - 1)
            ranges[a, m, 1] = min(int(min(hi, box) / voxel_size), grid_size - 1)
            m += 1
            if lo < 0.0:
                ranges[a, m, 0] = min(int((lo + box) / voxel_size), grid_size - 1)
                ranges[a, m, 1] = grid_size - 1
                m += 1
            if hi > box:
                ranges[a, m, 0] = 0
                ranges[a, m, 1] = min(int((hi - box) / voxel_size), grid_size - 1)
                m += 1
            n_ranges[a] = m

        for ri in range(n_ranges[0]):
            for vi in range(ranges[0, ri, 0], ranges[0, ri, 1] + 1):
                for rj in range(n_ranges[1]):
                    for vj in range(ranges[1, rj, 0], ranges[1, rj, 1] + 1):
                        for rk in range(n_ranges[2]):
                            for vk in range(ranges[2, rk, 0], ranges[2, rk, 1] + 1):
                                key = (vi * grid_size + vj) * grid_size + vk
                                for s in range(offsets[key], offsets[key + 1]):
                                    q = order[s]
                                    dx = coords[q, 0] - c[0]
                                    dy = coords[q, 1] - c[1]
                                    dz = coords[q, 2] - c[2]
                                    dx -= box * np.floor(dx / box + 0.5)
                                    dy -= box * np.floor(dy / box + 0.5)
                                    dz -= box * np.floor(dz / box + 0.5)
                                    if dx * dx + dy * dy + dz * dz <= R2:
                                        gas[h] += w_gas[q]
                                        dm[h] += w_dm[q]
                                        count[h] += 1
    return gas, dm, count


def Crossed_Halo_Gas(sightlines, snapshot, voxel_grid, w_gas, w_dm, box_size):
    """
    Gas within R200c of every halo crossed by a sightline in this snapshot, from the gas already in memory
    (voxel_grid from Build_Voxel_Grid over the same particles; weights from Halo_Gas_Weights). Each crossing's halo
    copy gets 'GasMass' = gas within R200c [Msun] (the catalogue/FoF value is kept as 'GasMassFoF'), 'GasMassDM' =
    DM-equivalent gas within R200c [Msun] and 'NGasR200c'. Halos that already have 'GasMassFoF' are skipped.
    Returns the number of distinct halos measured.
    """
    crossings = {}
    for sl in sightlines:
        for i in np.flatnonzero(np.asarray(sl.sub_Snapshots) == snapshot):
            for halo in sl.sub_Halos[i]:
                if halo is not None and 'GasMassFoF' not in halo:
                    crossings.setdefault(int(halo['ID']), []).append(halo)
    if not crossings:
        return 0

    ids = list(crossings)
    centres = np.array([crossings[i][0]['Pos'] for i in ids], dtype=np.float64)
    radii = np.array([crossings[i][0]['Radius'] for i in ids], dtype=np.float64)
    gas, dm, count = _Sum_Gas_In_Spheres(centres, radii, voxel_grid['coords'], w_gas, w_dm,
                                         voxel_grid['order'], voxel_grid['offsets'], voxel_grid['grid_size'],
                                         voxel_grid['voxel_size'], float(box_size))
    for k, hid in enumerate(ids):
        for halo in crossings[hid]:
            halo['GasMassFoF'] = halo['GasMass']
            halo['GasMass'] = float(gas[k] * 1e10)
            halo['GasMassDM'] = float(dm[k] * 1e10)
            halo['NGasR200c'] = int(count[k])
    return len(ids)


# --- Overarching Function -- #

def Points_In_Sightline(sightline,snapshot,architecture,radii,coarse_radius,findtype,giant_idx=None,giant_pts=None,giant_radii=None):
    
    # -- Iterate over sub sightlines, checking for those which coincide with the given snapshot number -- #
    for i in range(sightline.num_sub_sightlines):
        if sightline.sub_Snapshots[i] == snapshot:

            if len(sightline.sub_PointsIdx[i]) > 0:
                continue

            # -- Select between KDTree / Not -- #
            if findtype == 'tree':
                cylinder_idx = _Points_Near_Ray_Tree(architecture,sightline.sub_Origins[i],sightline.sub_Lengths[i],sightline.direction_vector,
                                                radii,coarse_radius,giant_idx,giant_pts,giant_radii)
                sightline.sub_PointsIdx[i] = cylinder_idx

            elif findtype == 'voxel':
                cylinder_idx = _Points_Near_Ray_Voxel(architecture,sightline.sub_Origins[i],sightline.sub_Lengths[i],sightline.direction_vector,
                                                radii,coarse_radius,giant_idx,giant_pts,giant_radii)
                sightline.sub_PointsIdx[i] = cylinder_idx

            else:
                SL = sightline.get_subsightline(i)

                c1,c2 = _Gen_Cross_Section(SL,np.nanmax(radii))
                limits = _Gen_Cubic_Volume_Limits(c1,c2)

                ts = clock()
                print('    points subset',end='\r')
                points_idx = _Points_Subset(limits,architecture)
                print(f'    points subset -- Done ({clock()-ts:.1f}s)')

                ts = clock()
                print('    points inside cylinder',end='\r')
                inside = _Points_Inside_Cylinder(architecture[points_idx],SL.origin,
                                              SL.transformation_matrix,
                                              SL.length,coarse_radius)
                print(f'    points inside cylinder -- Done ({clock()-ts:.1f}s)')

                ts = clock()
                print('    final filtering',end='\r')
                points_idx[points_idx==True] = np.logical_and(points_idx[points_idx==True],inside)
                print(f'    final filtering -- Done ({clock()-ts:.1f}s)')
                
                sightline.sub_PointsIdx[i] = np.where(points_idx)[0]

    return sightline






# -- Halo find functions -- #

# def _Halos_Near_Ray(sightline, halos):
#     """
#     halos: numpy array of dicts, each with keys 'Radius' and 'COM'
#     """

#     # Extract Radius and COM into arrays
#     radii = np.array([h['Radius'] for h in halos])
#     centre_of_mass = np.array([h['COM'] for h in halos])  # shape (N,3)

#     valid = radii > 0
#     radii = radii[valid]
#     centre_of_mass = centre_of_mass[valid]

#     halo_pos_vec = centre_of_mass - sightline.origin  # Vector from sightline origin to halos
#     dot = np.dot(halo_pos_vec, sightline.direction_vector)  # Dot product with sightline direction vector -> result is length along direction vector
#     projection = np.outer(dot, sightline.direction_vector)  #  # Projection vectors along sightline direction, shape (M,3)
#     impact_params = np.linalg.norm(projection - halo_pos_vec, axis=1)  # distance from halo to projection on sightline

    
#     # Repeat, but shift halo COM back along sightline by radius to see if sightline end just intersects halo
#     shifted_halo_pos_vec = centre_of_mass - sightline.origin - radii[:, None] * sightline.direction_vector  
#     shifted_dot = np.dot(shifted_halo_pos_vec, sightline.direction_vector)
#     shifted_proj = shifted_dot[:, None] * sightline.direction_vector
#     shifted_impactparam = np.linalg.norm(shifted_halo_pos_vec - shifted_proj, axis=1)

#     condition1 = (impact_params<radii) & (dot >= 0) & (dot <= sightline.length)
#     condition2 = (shifted_impactparam <= radii) & (shifted_dot >= 0) & (shifted_dot <= sightline.length)
        
#     intersects_mask = condition1 | condition2
#     valid_indices = np.where(valid)[0]  # indices of valid halos in original array
#     intersect_indices = valid_indices[intersects_mask]

#     # Build result list of halos with ImpactParam set
#     x_halos = []
#     for idx in intersect_indices:
#         halo = halos[idx].copy()  # copy dict to avoid modifying original
#         if condition1[intersects_mask][np.where(intersect_indices == idx)[0][0]]:
#             halo['ImpactParam'] = impact_params[intersects_mask][np.where(intersect_indices == idx)[0][0]]
#         else:
#             halo['ImpactParam'] = 'Maybe partial intersection.'
#         x_halos.append(halo)

#     return x_halos


@njit(cache=True, fastmath=True, nogil=True, parallel=True)
def _Halos_Near_Ray(origin, direction, length, com, radii):
    N = radii.shape[0]

    impact     = np.empty(N, dtype=np.float64)
    intersects = np.zeros(N, dtype=np.bool_)
    partial    = np.zeros(N, dtype=np.bool_)

    for i in prange(N):
        impact[i] = 0.0

        if radii[i] <= 0.0:
            continue

        hx = com[i, 0] - origin[0]
        hy = com[i, 1] - origin[1]
        hz = com[i, 2] - origin[2]

        dot = hx * direction[0] + hy * direction[1] + hz * direction[2]

        px = dot * direction[0]
        py = dot * direction[1]
        pz = dot * direction[2]

        dx = hx - px
        dy = hy - py
        dz = hz - pz

        impact_param = (dx*dx + dy*dy + dz*dz) ** 0.5
        impact[i]    = impact_param

        if impact_param < radii[i]:
            half_chord = (radii[i]**2 - impact_param**2) ** 0.5
            t_enter = dot - half_chord
            t_exit  = dot + half_chord

            intersects[i] = (t_exit >= 0.0) and (t_enter <= length)
            partial[i]    = intersects[i] and (t_enter < 0.0 or t_exit > length)

            if partial[i]:
                # Distance to closest point on the segment
                t_clamped = min(max(dot, 0.0), length)
                cx = hx - t_clamped * direction[0]
                cy = hy - t_clamped * direction[1]
                cz = hz - t_clamped * direction[2]
                impact[i] = (cx*cx + cy*cy + cz*cz) ** 0.5
            else:
                impact[i] = impact_param

    return intersects, partial, impact


def Halos_In_Sightline(sightline, snapshot, halos, com, radii, tree, max_radius,cosmo):

    from ._compute import Transform_Points
    from ._sightline_class import z_at_value
    import astropy.units as u

    for i in range(sightline.num_sub_sightlines):

        if sightline.sub_Snapshots[i] != snapshot:
            continue

        SL = sightline.get_subsightline(i)

        # Use existing gen_line() instead of _sample_ray
        sample_pts = SL.gen_line(n_samples=int(2*SL.length//max_radius))  # (n_samples, 3)

        raw_candidates = tree.query_ball_point(sample_pts, r=max_radius)
        candidates = np.unique(np.concatenate(raw_candidates)).astype(int) if any(raw_candidates) else np.array([], dtype=int)

        if len(candidates) == 0:
            sightline.sub_Halos[i] = [None]
            continue

        intersects, partial, impact = _Halos_Near_Ray(
            SL.origin,
            SL.direction_vector,
            SL.length,
            com[candidates],
            radii[candidates],
        )

        hit_local_indices = np.where(intersects)[0]

        if len(hit_local_indices) == 0:
            sightline.sub_Halos[i] = [None]
            continue

        prelength = np.nansum(sightline.sub_Lengths[:i])

        result = []
        for local_j in hit_local_indices:
            global_j  = candidates[local_j]
            halo_dict = dict(halos[global_j])
            halo_dict['ImpactParam']        = None if partial[local_j] else np.float32(impact[local_j])
            
            if halo_dict['ImpactParam'] == None and i == 0:
                redshift = 0
            else:
                halo_dist = Transform_Points(SL, halo_dict['Pos'])[2]
                redshift = z_at_value(cosmo.comoving_distance, (prelength + halo_dist) * u.kpc).value

            halo_dict['Redshift'] = np.float32(redshift)

            result.append(halo_dict)

        sightline.sub_Halos[i] = result

    return sightline

    





# def halos_near_ray_numba(sightline, halos):
#     radii = np.array([h['Radius'] for h in halos], dtype=np.float64)
#     com = np.array([h['COM'] for h in halos], dtype=np.float64)

#     origin = np.asarray(sightline.origin, dtype=np.float64)
#     direction = np.asarray(sightline.direction_vector, dtype=np.float64)
#     length = float(sightline.length)

#     intersects, partial, impact = halos_near_ray_numba_kernel(
#         origin, direction, length, com, radii
#     )

#     out = []
#     for i in np.where(intersects)[0]:
#         halo = halos[i].copy()
#         if not partial[i]:
#             halo['ImpactParam'] = impact[i]
#         else:
#             halo['ImpactParam'] = 'Maybe partial intersection.'
#         out.append(halo)

#     return out






# def _Iterate_subset_boxes(iterations,sightline,points):
#     """
#     Iterates over subboxes, returning a boolean array for points inside cylinder (TRUE) or not (FALSE).
#     """

#     all_points = np.zeros(len(points))
#     for i in range(iterations):
#         subLength = sightline.length/iterations
#         subOrigin = sightline.origin + i*subLength*sightline.direction_vector
#         subSightline = Sightline(radius=sightline.radius,direction_vector=sightline.direction_vector,length=subLength,origin=subOrigin)
#         c1,c2 = _Gen_cross_section(subSightline)
#         limits = _Gen_cubic_volume_limits(c1,c2)
#         all_points = np.logical_or(all_points,_Points_subset(limits,points))

#     return all_points

# def _Iterate_subset_boxes(iterations, sightline, points):
#     """
#     Efficiently iterates over subboxes along the cylinder axis and returns
#     a boolean mask for all points inside any subbox.
#     """
#     # Pre-allocate boolean array instead of float zeros
#     all_mask = np.zeros(points.shape[0], dtype=bool)

#     # Precompute constants
#     subLength = sightline.length / iterations
#     dir_vec = sightline.direction_vector
#     radius = sightline.radius

#     for i in range(iterations):
#         subOrigin = sightline.origin + i * subLength * dir_vec

#         # Generate subSightline values directly, avoid creating object
#         subSightline = Sightline(
#             radius=radius,
#             direction_vector=dir_vec,
#             length=subLength,
#             origin=subOrigin
#         )

#         # Use fast geometry methods
#         c1, c2 = _Gen_cross_section(subSightline)
#         limits = _Gen_cubic_volume_limits(c1, c2)

#         # Mask just this sub-box
#         mask = _Points_subset(limits, points)

#         # Combine masks in-place
#         np.logical_or(all_mask, mask, out=all_mask)

#     return all_mask

# @njit(parallel=True)
# def _Filter_points(pointsIdx, pointsIdx2):
#     j = 0
#     for i in prange(pointsIdx.shape[0]):
#         if pointsIdx[i]:
#             pointsIdx[i] = pointsIdx2[j]
#             j += 1
#     return pointsIdx






# @njit(parallel=True,fastmath=True)
# def _Refine_Points(ray_origin, ray_direction, ray_length, points, radii):

#     # -- Initialise -- #
#     num_points = points.shape[0]
#     mask = np.zeros(num_points, dtype=np.bool_)

#     rx, ry, rz = ray_direction
#     ox, oy, oz = ray_origin

#     # -- Iterate over points -- #
#     for i in prange(num_points):
#         # Point vector
#         px = points[i, 0] - ox
#         py = points[i, 1] - oy
#         pz = points[i, 2] - oz
        
#         # Dot product for projection
#         t = px * rx + py * ry + pz * rz
        
#         # Clamp t to [0, ray_len]
#         if t < 0.0: t = 0.0
#         elif t > ray_length: t = ray_length

#         # Closest point on ray segment
#         cpx, cpy, cpz = ox + t * rx, oy + t * ry, oz + t * rz

#         # Squared Distance calculation
#         dx, dy, dz = points[i, 0] - cpx, points[i, 1] - cpy, points[i, 2] - cpz
#         dist2 = dx*dx + dy*dy + dz*dz

#         if dist2 <= radii[i]**2:
#             mask[i] = True

#     return mask