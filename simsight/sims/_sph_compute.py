import numpy as np
from numba import njit
from scipy.spatial import cKDTree

@njit(fastmath=True)
def Find_Intersection_Intervals(points, smoothing_length):
    """
    Finds the parametric t value where the ray enters and exits the kernel of each particle.
    Returns a numpy array where 
        - intervals[:,0] are the point idx,
        - intervals[:,1] are the t1
        - intervals[:,2] are the t2
    """

    n_particles = points.shape[0]
    # Pre-allocate
    intervals = np.empty((n_particles, 3), dtype=np.float32)
    cursor = 0

    for i in range(n_particles):
        # Accessing elements directly is faster than creating sub-arrays
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        
        # Rad = h * 2
        r = smoothing_length[i] * 2.0
        
        # Simplified discriminant for ray at (0,0,0) pointing (0,0,1)
        # The intersection only depends on the X and Y distance from the axis
        disc = (r * r) - (px * px + py * py)
        
        if disc >= 0.0:
            sqrt_disc = np.sqrt(disc)
            intervals[cursor, 0] = i
            # t = z +/- sqrt_disc
            intervals[cursor, 1] = pz - sqrt_disc
            intervals[cursor, 2] = pz + sqrt_disc
            cursor += 1

    return intervals[:cursor]


@njit(inline='always')
def _Evaluate_Kernel(q, h, kernel_type):
    """
    Returns (weight) for a given q = r/h.
    kernel_type: 0 = Wendland C2, 1 = Cubic Spline
    """
    inv_h = 1.0 / h
    
    if kernel_type == 0:  # Wendland C2
        # Norm for 3D: 21 / (16 * pi * h^3)
        norm = (21.0 / (16.0 * np.pi)) * (inv_h**3)
        return norm * (1.0 - 0.5*q)**4 * (2.0*q + 1.0)
    
    elif kernel_type == 1:  # Cubic Spline (M4)
        # Norm for 3D: 1 / (pi * h^3)
        norm = (1.0 / (np.pi * h**3))
        if q < 1.0:
            return norm * (1.0 - 1.5*q**2 + 0.75*q**3)
        else: # 1 <= q < 2
            return norm * (0.25 * (2.0 - q)**3)
            
    return 0.0

@njit(fastmath=True, nogil=True)
def Build_Sparse_Weights(t_grid, points, smooth_lengths, p_idx, p_z1, p_z2, kernel_type):
    """
    Build three 1-d arrays which include the pointIdx, the gridIdx, and the resulting weight.
    e.g. the nth entry indicates that particle p[n] contributes a weight w[n] and grid point i[n]
    """

    n_intervals = p_idx.shape[0]
    
    # --- PASS 1: COUNT ENTRIES ---
    total_entries = 0
    for k in range(n_intervals):
        idx = p_idx[k]
        i1 = np.searchsorted(t_grid, p_z1[k])
        i2 = np.searchsorted(t_grid, p_z2[k])
        
        px, py, pz = points[idx, 0], points[idx, 1], points[idx, 2]
        inv_h2 = 1.0 / (smooth_lengths[idx]**2)
        r_xy2 = px*px + py*py

        for i in range(i1, i2):
            dz = t_grid[i] - pz
            q2 = (r_xy2 + dz*dz) * inv_h2
            if q2 < 4.0: # r < 2h
                total_entries += 1

    # --- ALLOCATE EXACT SIZED ARRAYS ---
    out_p = np.empty(total_entries, dtype=np.int64)
    out_i = np.empty(total_entries, dtype=np.int64)
    out_w = np.empty(total_entries, dtype=np.float64)

    # --- PASS 2: FILL ARRAYS ---
    cursor = 0
    for k in range(n_intervals):
        idx = p_idx[k]
        i1 = np.searchsorted(t_grid, p_z1[k])
        i2 = np.searchsorted(t_grid, p_z2[k])
        
        px, py, pz = points[idx, 0], points[idx, 1], points[idx, 2]
        h = smooth_lengths[idx]
        inv_h2 = 1.0 / (smooth_lengths[idx]**2)
        r_xy2 = px*px + py*py

        for i in range(i1, i2):
            dz = t_grid[i] - pz
            q2 = (r_xy2 + dz*dz) * inv_h2
            
            if q2 < 4.0:
                q = np.sqrt(q2)
                out_p[cursor] = idx
                out_i[cursor] = i
                out_w[cursor] = _Evaluate_Kernel(q, h, kernel_type)
                cursor += 1

    return out_p[:cursor], out_i[:cursor], out_w[:cursor]


@njit(fastmath=True)
def Calculate_Field(result_array, out_p, out_i, out_w, masses, volumes, scalar_field,  is_sph):
    """
    Compute interpolated quantities at arbitrary points using MFM kernel with normalization.

    Parameters
    ----------
    out_p : 1D int array
        Particle indices corresponding to the neighbor list for all points.
    out_i : 1D int array
        Output point indices (which arbitrary point each weight belongs to).
    out_w : 1D float array
        Kernel weights W(|r_p - r_j|, h_j) for each particle-point pair.
    masses : 1D float array
        Mass of each particle.
    density : 1D float array
        Density of each particle (rho_j = m_j / V_j).
    scalar_field : 1D float array or None
        Field to interpolate (e.g., metallicity, SFR). If None, function computes density.
    array : 1D float array
        Array to store interpolated results per point (same length as number of arbitrary points).
    """

    # Temporary arrays to accumulate numerator and denominator
    n_points = np.max(out_i) + 1
    numerator = np.zeros(n_points, dtype=np.float64)
    denominator = np.zeros(n_points, dtype=np.float64)

    for k in range(out_w.shape[0]):
        points_idx = out_p[k]           # particle index
        t_idx = out_i[k]           # arbitrary point index
        weight = out_w[k]              # kernel weight

        if scalar_field is None:
            # For density, numerator is m_j * W_j
            numerator[t_idx] += masses[points_idx] * weight
        else:
            # For other scalar fields, numerator is A_j * V_j * W_j
            numerator[t_idx] += scalar_field[points_idx] * volumes[points_idx] * weight

        # Denominator is sum of V_j * W_j (normalization)
        denominator[t_idx] += volumes[points_idx] * weight

    # Final result: numerator / denominator
    for i in range(n_points):
        if scalar_field is None and is_sph:
            # PURE SPH DENSITY: Just the raw summation
            result_array[i] = numerator[i]
        else:
            # MFM (SIMBA) or SPH SCALARS: Requires normalization
            if denominator[i] > 1e-40:
                result_array[i] = numerator[i] / denominator[i]
            else:
                result_array[i] = 0.0
        

def Nearest_Particle(t_grid,coordinates):

    grid_points = np.zeros((len(t_grid), 3))
    grid_points[:, 2] = t_grid

    tree = cKDTree(coordinates)
    _, nearest_idx = tree.query(grid_points, k=1)

    return nearest_idx