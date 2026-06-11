def load_sim(data_path,snap_path_structure=None,halo_path_structure=None,fsps_path=None):
    if 'TNG' in data_path:
        from .tng import TNG_SightlineSim
        return TNG_SightlineSim(data_path,fsps_path=fsps_path)
    elif 'SIMBA' in data_path:
        from .simba import SIMBA_SightlineSim
        return SIMBA_SightlineSim(data_path,snap_path_structure,halo_path_structure,fsps_path=fsps_path)
    else:
        raise ValueError("Could not determine simulation type from data_path.")