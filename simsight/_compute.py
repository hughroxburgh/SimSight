import numpy as np
import matplotlib.pyplot as plt

def Transform_Points(sightline, points, inverse=False):

    # Rearrange so basis order is v2, v3, v1 → X, Y, Z
    tm = sightline.transformation_matrix[:, [1, 2, 0]]

    if inverse:
        # Transform from basis coords back to Cartesian
        return (points @ tm.T) + sightline.origin
    else:
        # Transform from Cartesian to basis coords
        return (points - sightline.origin) @ tm


def Compute_Sightline(sightline,sim,data,func,snapshot):
    """
    Compute a desired function along the portion (sub_sightlines) that exist in this snapshot.
    """

    results = []

    # -- Iterate over sub sightlines -- #
    for i in range(sightline.num_sub_sightlines):
        if sightline.sub_Snapshots[i] == snapshot:  # check if sub sightline in this snapshot
            
            if len(sightline.sub_Compute[i]) > 0:
                continue
            
            idx = sightline.sub_PointsIdx[i]    # sub sightline idx pointing to which points matter

            if len(idx) > 0:

                #  Generate a Sightline object to perform basis transformation
                subSL = sightline.get_subsightline(idx=i)   
                transformed_points = Transform_Points(subSL, data['Coordinates'][idx])

                # Isolate data that is needed for this sightline
                used_fields = {k: v[idx] for k, v in data.items() if k != 'Coordinates'}

                # Generate field profiles along the ray
                ray_data, lengths, ids = sim.process_data(transformed_points,used_fields,sightline.sub_Lengths[i])

                # Compute the function
                computed_array = func(ray_data,lengths,sightline.sub_BoxRedshifts[i])

                results.append((i,computed_array,ray_data['Density'],lengths,idx[ids]))

    return results


# --------- Computable Functions --------- #

def Calc_Ray_Density(ray_data,lengths,redshift):

    return ray_data['Density'] * 10**10   # Msun / ckpc^2 

def Calc_Ray_DM(ray_data,lengths,redshift):
    """
    A ray's DM is equal to the integral of the physical free electron density / 1+z through the physical distance traversed by the ray.
    z in this case is the redshift of the snapshot the ray travels through. This 1+z arises due to a combination of cosmological redshift 
    acting on the plasma frequency of the light, as well as the time dilation conversion from proper time to observed time.

    In Sims, we use comoving coordinates, so we need to make more transformations. dl_phys = dl_com / 1+z, so this adds in a second factor
    to the denominator. Then, the physical free electron density is equal to the comoving free electron density multiplied by (1+z)^3, 
    because as we go back in redshift, the three spatial dimensions are smaller, so the physical number density increases.

    We now have DM_i = integral 0->L  of (1+z)^3 ne_com / (1+z)^2 dl_com     =     integral 0->L of (1+z)*ne_com*dl_com.
    We can get ne_com easily.
    """

    densities = ray_data['Density']         # 10^10 Msun / ckpc^3
    electronAbundance = ray_data['ElectronAbundance']
    starFormationRates = ray_data['StarFormationRate']
    w = (starFormationRates==0).astype(int)

    xH = 0.76 # hydrogenFraction
    mP = 1.67e-24 # protonMass in g
    constant = 1.988e43 / (3.086e21 ** 3)   # 10^10 Msun -> g / 1kpc^3 -> cm^3

    freeElectronDensity_com =  w * electronAbundance * densities * xH/mP * constant  # in cm^-3  
    freeElectronDensity_phys = (1+redshift)**3 * freeElectronDensity_com
    DM = freeElectronDensity_phys * (lengths*1000) / (1+redshift)**2

    return DM  
