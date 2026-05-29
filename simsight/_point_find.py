import numpy as np
from numba import njit,prange
from time import time

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

def _Points_Near_Ray(tree, ray_origin, ray_length, ray_direction, radii, coarse_radius,giant_idx,giant_pts,giant_radii):
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



# --- Overarching Function -- #

def Points_In_Sightline(sightline,snapshot,tree_or_points,radii,coarse_radius,giant_idx=None,giant_pts=None,giant_radii=None,treebased=True):
    
    # -- Iterate over sub sightlines, checking for those which coincide with the given snapshot number -- #
    for i in range(sightline.num_sub_sightlines):
        if sightline.sub_Snapshots[i] == snapshot:

            if len(sightline.sub_PointsIdx[i]) > 0:
                continue

            # -- Select between KDTree / Not -- #
            if treebased:
                cylinder_idx = _Points_Near_Ray(tree_or_points,sightline.sub_Origins[i],sightline.sub_Lengths[i],sightline.direction_vector,
                                                radii,coarse_radius,giant_idx,giant_pts,giant_radii)
                sightline.sub_PointsIdx[i] = cylinder_idx

            else:
                SL = sightline.get_subsightline(i)

                c1,c2 = _Gen_Cross_Section(SL,np.nanmax(radii))
                limits = _Gen_Cubic_Volume_Limits(c1,c2)

                ts = time()
                print('    points subset',end='\r')
                points_idx = _Points_Subset(limits,tree_or_points)
                print(f'    points subset -- Done ({time()-ts:.1f}s)')

                ts = time()
                print('    points inside cylinder',end='\r')
                inside = _Points_Inside_Cylinder(tree_or_points[points_idx],SL.origin,
                                              SL.transformation_matrix,
                                              SL.length,coarse_radius)
                print(f'    points inside cylinder -- Done ({time()-ts:.1f}s)')

                ts = time()
                print('    final filtering',end='\r')
                points_idx[points_idx==True] = np.logical_and(points_idx[points_idx==True],inside)
                print(f'    final filtering -- Done ({time()-ts:.1f}s)')
                
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


@njit(cache=True, fastmath=True)
def _Halos_Near_Ray(origin, direction, length, com, radii):
    N = radii.shape[0]

    impact = np.empty(N, dtype=np.float64)
    intersects = np.zeros(N, dtype=np.bool_)
    partial = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        if radii[i] <= 0.0:
            impact[i] = 0.0
            continue

        # halo_vec = com[i] - origin
        hx = com[i, 0] - origin[0]
        hy = com[i, 1] - origin[1]
        hz = com[i, 2] - origin[2]

        dot = hx * direction[0] + hy * direction[1] + hz * direction[2]

        # proj = dot * direction
        px = dot * direction[0]
        py = dot * direction[1]
        pz = dot * direction[2]

        dx = hx - px
        dy = hy - py
        dz = hz - pz

        impact_param = (dx*dx + dy*dy + dz*dz) ** 0.5

        c1 = (impact_param < radii[i]) and (dot >= 0.0) and (dot <= length)

        # shifted halo
        shx = hx - radii[i] * direction[0]
        shy = hy - radii[i] * direction[1]
        shz = hz - radii[i] * direction[2]

        sdot = shx * direction[0] + shy * direction[1] + shz * direction[2]

        spx = sdot * direction[0]
        spy = sdot * direction[1]
        spz = sdot * direction[2]

        sdx = shx - spx
        sdy = shy - spy
        sdz = shz - spz

        shifted_impact = (sdx*sdx + sdy*sdy + sdz*sdz) ** 0.5

        c2 = (shifted_impact <= radii[i]) and (sdot >= 0.0) and (sdot <= length)

        intersects[i] = c1 or c2
        partial[i] = (not c1) and c2
        impact[i] = impact_param

    return intersects, partial, impact


def Halos_In_Sightline(sightline,snapshot,halos,com,radii):
    
    # -- Iterate over sub sightlines, checking for those which coincide with the given snapshot number -- #
    for i in range(sightline.num_sub_sightlines):
        if sightline.sub_Snapshots[i] == snapshot:
            SL = sightline.get_subsightline(i)

            intersects, partial, impact = _Halos_Near_Ray(SL.origin,SL.direction_vector,SL.length,com,radii)

            if len(np.where(intersects)[0]) > 0:
                x_halos = []
                for j in np.where(intersects)[0]:
                    halo = halos[j].copy()
                    if not partial[j]:
                        halo['ImpactParam'] = impact[j]
                    else:
                        halo['ImpactParam'] = 'Maybe partial intersection.'
                    x_halos.append(halo)

                sightline.sub_Halos[i] = x_halos
            else:
                sightline.sub_Halos[i] = [None]

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