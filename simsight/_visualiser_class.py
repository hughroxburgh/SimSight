import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from contextlib import contextmanager

from tqdm import tqdm
import imageio
import io
from ._utils import _Get_Colours


class VisualSim():

    def __init__(self,parent,dark_mode=False):

        self.parent = parent
        self.sim = self.parent.sim

        self.fig = None
        self.ax = None

        self.dark_mode = dark_mode

    @contextmanager
    def _style(self):
        style = 'dark_background' if self.dark_mode else 'default'
        with plt.style.context(style):
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 14,
                'axes.labelsize': 14,
                'axes.titlesize': 15,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 12,
            })
            yield

    def _resolve_filter(self, sightlines, filt=None, sweep_param=None, sweep_values=None, cmap='viridis'):
        base_kwargs = dict(filt) if filt is not None else {}

        if sweep_param is None:
            if filt is None:
                return [(None, sightlines, None)]
            mask = self.parent.filter_sightlines(sightlines, **base_kwargs)
            return [(None, sightlines[mask], None)]

        if sweep_values is None or len(sweep_values) == 0:
            raise ValueError("sweep_values must be provided when sweep_param is set")

        cmap_obj = plt.get_cmap(cmap)
        n = len(sweep_values)
        results = []
        for i, val in enumerate(sweep_values):
            kwargs = dict(base_kwargs)
            kwargs[sweep_param] = val
            mask = self.parent.filter_sightlines(sightlines, **kwargs)
            color = cmap_obj(i / max(n - 1, 1))

            if isinstance(val, (int, float)) and abs(val) >= 100000:
                exponent = int(np.floor(np.log10(abs(val))))
                mantissa = val / 10**exponent
                if abs(mantissa) >= 10:
                    mantissa /= 10
                    exponent += 1
                mantissa_str = f'{int(mantissa)}' if mantissa == int(mantissa) else f'{mantissa:.2g}'
                val_str = rf'${mantissa_str}\times10^{{{exponent}}}$'
            else:
                val_str = str(val)

            results.append((f'{sweep_param}={val_str}', sightlines[mask], color))
        return results

    # ------------- miscellaneous functions ------------- #

    def plot3d(self):
        """
        Generates a 3D MPL box for plotting.
        """

        with self._style():
            # fig, ax = plt.subplots()

            self.fig = plt.figure(figsize=(10,10))
            self.ax = self.fig.add_subplot(projection='3d')
            self.ax.set_xlim3d(0,self.sim.box_size)
            self.ax.set_ylim3d(0,self.sim.box_size)
            self.ax.set_zlim3d(0,self.sim.box_size)
            self.ax.set_xlabel('X [kpc]')
            self.ax.set_ylabel('Y [kpc]')
            self.ax.set_zlabel('Z [kpc]') 

    # ------------- Plotting sightlines ------------- #

    def plot_sightline(self,sightline,points=None,n_subsightlines='all',colour=None,halos=True):

        if n_subsightlines == 'all':
            n_subsightlines = sightline.num_sub_sightlines

        if colour is None:
            cmap = _Get_Colours(n_subsightlines,dark_mode=self.dark_mode)
        else:
            if isinstance(colour, str):
                cmap = np.full(n_subsightlines, colour, dtype=object) 
            else:
                cmap = np.empty(n_subsightlines, dtype=object)
                cmap[:] = [colour]

        if points is not None:
            for i,idx in enumerate(sightline.sub_PointsIdx[:n_subsightlines]):
                self.ax.scatter(points[idx][:,0],
                                points[idx][:,1],
                                points[idx][:,2],s=10,depthshade=True,alpha = 0.4,color=cmap[i])
                
        else:
            for i in range(n_subsightlines):
                origin = sightline.sub_Origins[i]
                ray = sightline.gen_line(subsightline=i)
                if colour is not None:
                    if i == 0:
                        self.ax.scatter(origin[0],origin[1],origin[2],s=10,facecolor=cmap[i],edgecolor='black' if not self.dark_mode else 'white')
                else:
                    self.ax.scatter(origin[0],origin[1],origin[2],s=10,facecolor=cmap[i],edgecolor='black' if not self.dark_mode else 'white')
                self.ax.plot(ray[:,0],ray[:,1],ray[:,2],color=cmap[i])

        if halos:

            theta = np.linspace(0,np.pi,100)
            phi = np.linspace(0,2*np.pi,100)
            theta,phi = np.meshgrid(theta,phi)
            for i,halos in enumerate(sightline.sub_Halos[:n_subsightlines]):
                for halo in halos:
                    if halo is not None:
                        x = halo['Pos'][0]+halo['Radius']*np.sin(theta)*np.cos(phi)
                        y = halo['Pos'][1]+halo['Radius']*np.sin(theta)*np.sin(phi)
                        z = halo['Pos'][2]+halo['Radius']*np.cos(theta)
                        self.ax.plot_surface(x,y,z,color=cmap[i],alpha=0.8)

    def plot_many_sightlines(self,sightlines,n_sightlines,points=None,n_subsightlines='all',halos=False):

        idx = np.random.choice(len(sightlines),size=n_sightlines, replace=False)

        colours = _Get_Colours(n_sightlines,dark_mode=self.dark_mode)

        self.plot3d()
        for ii,idx in enumerate(idx):
            self.plot_sightline(sightlines[idx],points,n_subsightlines,colour=colours[ii],halos=halos) 



    # ------------- Data results visualisation ------------- #

    def fullsky_image(self,sightlines,functype='DM',cmap=None,colour='dodgerblue',cutoff=99.5,
                      redshift=None,gif_path=None,environment='Total',modelled=False,fgas=None,figm=None):

        import healpy as hp


        if cmap is None:
            colors = ["black", colour, "white"]
            cmap = mcolors.LinearSegmentedColormap.from_list("black_colour_white", colors)

        dvecs = np.array([sl.direction_vector for sl in sightlines])

        max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment=environment,modelled=modelled)
        redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
        redshifts = [z for z in redshift_input if z <= max_redshift]
        if max(redshift_input) > max_redshift:
            redshifts.append(max_redshift)

        data_source = 'truth' if not modelled else 'modelled'
        vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled,fgas,figm) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} {environment} compute')])
        # if vals.ndim == 1:
        #     vals = vals[:, np.newaxis]
            
        npix = len(sightlines)

        try:
            nside = hp.npix2nside(npix)
        except:
            raise ValueError('Sightline number not drawn from healpix')

        ipix = hp.vec2pix(nside, dvecs[:,0], dvecs[:,1], dvecs[:,2])

        frames = []
        for i in tqdm(range(len(redshifts))):
            z = redshifts[i]
            vs = vals[:,i]

            m = np.full(npix, hp.UNSEEN, dtype=float)
            acc = np.zeros(npix, dtype=float)
            cnt = np.zeros(npix, dtype=int)
            np.add.at(acc, ipix, vs)
            np.add.at(cnt, ipix, 1)
            mask = cnt > 0
            m[mask] = acc[mask] / cnt[mask]

            maxval = np.percentile(vs, cutoff)

            with self._style():
                hp.mollview(m, unit=functype, title=f'{self.sim.name} FullSky {functype} Map to z = {z:.2g}', 
                            badcolor="lightgray", cmap=cmap, max=maxval, min=0)

                if gif_path is not None:
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    frames.append(imageio.imread(buf))
                    buf.close()
                    plt.close()
                else:
                    plt.show()

        if gif_path is not None:
            imageio.mimsave(f'{gif_path}/{functype}_fullsky.gif', frames, duration=5000/len(frames))





    def distribution(self, sightlines, functype='DM', cutoff=98, bins=100, redshift=None, xlims=None,
                    gif_path=None, environment='Total', data='truth',fgas=None,figm=None,
                    filt=None, sweep_param=None, sweep_values=None, sweep_cmap='viridis'):

        colours = {'Total':_Get_Colours(3,self.dark_mode)[0],'CGM':_Get_Colours(3,self.dark_mode)[1],'IGM':_Get_Colours(3,self.dark_mode)[2]}

        max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment=environment, modelled='model' in data)
        redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
        redshifts = [z for z in redshift_input if z <= max_redshift]
        if max(redshift_input) > max_redshift:
            redshifts.append(max_redshift)

        groups = self._resolve_filter(sightlines, filt=filt, sweep_param=sweep_param,
                                    sweep_values=sweep_values, cmap=sweep_cmap)
        sweeping = sweep_param is not None

        # Precompute vals per group per data_source: {group_label: {'Truth': arr, 'Model': arr}}
        group_data = {}
        for label, group_sightlines, sweep_color in groups:
            if len(group_sightlines) == 0:
                print(f'Skipping {label}: filter left 0 sightlines')
                continue
            entry = {}
            if 'truth' in data:
                entry['Truth'] = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled=False)
                                            for sl in tqdm(group_sightlines, desc=f'Getting Truth {environment} Compute'
                                                        + (f' [{label}]' if label else ''))])
            if 'model' in data:
                entry['Model'] = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled=True,fgas=fgas,figm=figm)
                                            for sl in tqdm(group_sightlines, desc=f'Getting Modelled {environment} Compute'
                                                        + (f' [{label}]' if label else ''))])
            group_data[label] = (entry, sweep_color)

        frames = []
        for i in tqdm(range(len(redshifts))):
            z = redshifts[i]

            with self._style():

                fig,ax = plt.subplots()
                ax.set_title(f'{self.sim.name} {environment} {functype} Distribution to z = {z:.2g}')
                ax.set_xlabel(functype)
                ax.set_ylabel('Probability Density')

                legend_handles = []

                for label, (entry, sweep_color) in group_data.items():
                    for source_name, vals in entry.items():
                        vs = vals[:,i]
                        maxval = np.percentile(vs, cutoff)
                        color = sweep_color if sweeping else colours[environment]
                        vs_clipped = vs[vs < maxval]

                        if source_name == 'Truth':
                            ax.hist(vs_clipped, bins=bins, density=True, color=color, alpha=0.5, histtype='bar')
                        else:
                            ax.hist(vs_clipped, bins=bins, density=True, color=color, linewidth=2,
                                    linestyle='--', histtype='step')

                if sweeping:
                    for label, (_, sweep_color) in group_data.items():
                        legend_handles.append(Line2D([0], [0], color=sweep_color, linestyle='-', label=label))
                else:
                    legend_handles.append(Patch(facecolor=colours[environment], alpha=0.5, label=environment))

                if 'truth' in data:
                    legend_handles.append(Patch(facecolor='gray', alpha=0.5, label='Truth'))
                if 'model' in data:
                    legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white',
                                                linestyle='--', label='Model'))

                ax.legend(handles=legend_handles)

                if xlims is not None:
                    ax.set_xlim(xlims[0],xlims[1])

                if gif_path is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    frames.append(imageio.imread(buf))
                    buf.close()
                    plt.close(fig)
                else:
                    plt.show()

        if gif_path is not None:
            imageio.mimsave(f'{gif_path}/{functype}_distribution.gif', frames, duration=5000/len(frames))


    def cumulutive_stats(self, sightlines, stat='mean', functype='DM', bins=10, redshift=None,
                      yscale='linear', environment='Total', data='truth',fgas=None,figm=None,
                      filt=None, sweep_param=None, sweep_values=None, sweep_cmap='viridis'):

        if sweep_param is not None and environment == 'separate':
            raise ValueError("environment='separate' can't be combined with sweep_param "
                            "(both use color to encode different things) — pick one environment.")

        colours = {'Total':_Get_Colours(3,self.dark_mode)[0],'CGM':_Get_Colours(3,self.dark_mode)[1],'IGM':_Get_Colours(3,self.dark_mode)[2]}
        if environment == 'separate':
            environments = ['Total','CGM','IGM']
            check_env = 'CGM'
        else:
            environments = [environment]
            check_env = environment

        truth = 'truth' in data
        modelled = 'model' in data

        stat_map = {'mean': {'func':np.nanmean, 'name': 'Mean'},
                    'std' : {'func':np.nanstd, 'name': 'Standard Deviation'}}

        groups = self._resolve_filter(sightlines, filt=filt, sweep_param=sweep_param,
                                    sweep_values=sweep_values, cmap=sweep_cmap)
        sweeping = sweep_param is not None

        with self._style():

            fig,ax = plt.subplots()
            ax.set_xlabel('Redshift')
            ax.set_ylabel(f'Cumulutive {stat_map[stat]["name"]} {functype}')
            ax.set_title(f'{self.sim.name} {stat_map[stat]["name"]} {functype}')
            if yscale == 'log':
                ax.set_yscale('log')

            legend_handles = []
            if not sweeping:
                for env in environments:
                    legend_handles.append(Line2D([0], [0], color=colours[env], marker='x', linestyle='none', label=env))

            for label, group_sightlines, sweep_color in groups:

                if len(group_sightlines) == 0:
                    print(f'Skipping {label}: filter left 0 sightlines')
                    continue

                if truth:
                    r = np.nanmin([group_sightlines[0].redshift_reached(self.sim.cosmo, environment=check_env),
                                np.nan if redshift is None else redshift])
                    redshifts = np.linspace(0, r, bins)

                    for env in environments:
                        vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,env,modelled=False)
                                        for sl in tqdm(group_sightlines, desc=f'Getting Truth {env} Compute'
                                                        + (f' [{label}]' if label else ''))])
                        func = stat_map[stat]['func']
                        stats = func(vals, axis=0)
                        line_color = sweep_color if sweeping else colours[env]
                        ax.plot(redshifts, stats, 'x-', color=line_color)

                if modelled:
                    r = np.nanmin([group_sightlines[0].redshift_reached(self.sim.cosmo, environment=check_env, modelled=True),
                                np.nan if redshift is None else redshift])
                    redshifts = np.linspace(0, r, bins)

                    for env in environments:
                        vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,env,modelled=True,fgas=fgas,figm=figm)
                                        for sl in tqdm(group_sightlines, desc=f'Getting Modelled {env} Compute'
                                                        + (f' [{label}]' if label else ''))])
                        func = stat_map[stat]['func']
                        stats = func(vals, axis=0)
                        line_color = sweep_color if sweeping else colours[env]
                        ax.plot(redshifts, stats, 'x--', color=line_color)

                if sweeping:
                    legend_handles.append(Line2D([0], [0], color=sweep_color, linestyle='-', label=label))

            if truth and not sweeping:
                legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white', linestyle='-',  label='Truth'))
            if modelled and not sweeping:
                legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white', linestyle='--', label='Model'))

            ax.legend(handles=legend_handles)

            plt.show()


    def halo_partition(self, sightlines, functype='DM', cutoff=98, redshift=None, plottype='hist',
                        gif_path=None, modelled=False, filt=None,fgas=None,figm=None):

        if filt is not None:
            mask = self.parent.filter_sightlines(sightlines, **filt)
            sightlines = sightlines[mask]

        max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment='IGM', modelled=modelled)
        redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
        redshifts = [z for z in redshift_input if z <= max_redshift]
        if max(redshift_input) > max_redshift:
            redshifts.append(max_redshift)

        colours = {'CGM':_Get_Colours(3,self.dark_mode)[1],'IGM':_Get_Colours(3,self.dark_mode)[2]}

        data_source = 'truth' if not modelled else 'modelled'
        dms_cgm = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment='CGM',modelled=modelled,fgas=fgas,figm=figm) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} CGM compute')])
        dms_igm = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment='IGM',modelled=modelled,fgas=fgas,figm=figm) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} IGM compute')])

        order = np.argsort(dms_cgm[:, -1]+dms_igm[:, -1])

        maxval = np.percentile(dms_cgm[:, -1]+dms_igm[:, -1], cutoff)

        frames = []
        for i in tqdm(range(len(redshifts))):
            z = redshifts[i]
            dm_cgm = dms_cgm[:,i]
            dm_igm = dms_igm[:,i]

            dm_tot = dm_cgm + dm_igm

            with self._style():
                if plottype == 'hist':
                    x = np.arange(len(dm_tot))

                    fig, ax = plt.subplots()
                    ax.fill_between(x, 0, dm_igm[order], label='IGM',color=colours['IGM'])
                    ax.fill_between(x, dm_igm[order], dm_igm[order] + dm_cgm[order], label='CGM',color=colours['CGM'])

                    ax.set_title(f'{self.sim.name} {functype} Partition to z = {z:.2g}')
                    ax.set_xlabel('Sightlines')
                    ax.set_ylabel(f'{functype}')
                    ax.set_ylim(0, maxval)
                    ax.legend()

                elif plottype == 'scatter':
                    frac_cgm = dm_cgm / dm_tot
                    frac_igm = dm_igm / dm_tot

                    fig, ax = plt.subplots()
                    ax.scatter(dm_tot, frac_igm, s=10, alpha=0.6, label='IGM fraction',c=colours['IGM'])
                    ax.scatter(dm_tot, frac_cgm, s=10, alpha=0.6, label='CGM fraction',c=colours['CGM'])

                    ax.set_title(f'{self.sim.name} {functype} Partition to z = {z:.2g}')
                    ax.set_xlabel(f'Total {functype}')
                    ax.set_ylabel(f'Fractional {functype} Contribution')
                    ax.legend()

                if gif_path is not None:

                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    frames.append(imageio.imread(buf))
                    buf.close()
                    plt.close(fig)
                else:
                    plt.show()

        if gif_path is not None:
            imageio.mimsave(f'{gif_path}/{functype}_partition.gif', frames, duration=5000/len(frames))

    
    def halo_visibility(self, sightlines, parameter='redshift', parameter2=None,
                     plottype='fraction', entity='halo', weight='count',
                     status_only=None, bins=10, filt=None, sweep_param=None,
                     sweep_values=None, sweep_cmap='viridis',
                     xscale='linear', yscale='linear'):


        if entity not in ('galaxy', 'halo'):
            raise ValueError(f"entity must be 'galaxy' or 'halo', got '{entity}'")
        if entity == 'galaxy' and (weight == 'compute' or parameter.lower() == 'compute' or (parameter2 is not None and parameter2.lower() == 'compute')):
            raise ValueError("entity='galaxy' can't be combined with weight/parameter='compute' "
                            "(DM/Compute is only tracked per-halo, not per-galaxy) — "
                            "use entity='halo' for DM-weighted fractions, or weight='count' for galaxies.")
        if status_only is not None and status_only not in (1, 0, -1):
            raise ValueError(f"status_only must be None, 1, 0, or -1, got '{status_only}'")
        if sweep_param is not None and (sweep_values is None or len(sweep_values) < 2):
            raise ValueError("sweep_param requires sweep_values with at least 2 edges "
                            "(defining len(sweep_values)-1 groups), e.g. sweep_values=[1e9, 1e10, 1e11]")
        if plottype == 'hist' and sweep_param is not None and status_only is None:
            raise ValueError("plottype='hist' with sweep_param requires status_only to be set "
                            "(sweep colour and status colour can't both be encoded at once).")

        FILT_KEYS = {
            'min_redshift':      lambda h, g, v: h['Redshift'] >= v,
            'max_redshift':      lambda h, g, v: h['Redshift'] <= v,
            'min_halo_mass':     lambda h, g, v: h['TotalMass'] >= v,
            'max_halo_mass':     lambda h, g, v: h['TotalMass'] <= v,
            'min_stellar_mass':  lambda h, g, v: h['StellarMass'] >= v,
            'max_stellar_mass':  lambda h, g, v: h['StellarMass'] <= v,
            'min_impact_param':  lambda h, g, v: h['ImpactParam'] >= v,
            'max_impact_param':  lambda h, g, v: h['ImpactParam'] <= v,
            'min_gal_mass':      lambda h, g, v: g is not None and g['StellarMass'] >= v,
            'max_gal_mass':      lambda h, g, v: g is not None and g['StellarMass'] <= v,
        }

        if filt is not None and not isinstance(filt, dict):
            raise ValueError("filt must be a dict of {key: value}, e.g. {'max_halo_mass': 1e11}. "
                            f"Valid keys: {list(FILT_KEYS)}")
        if filt is not None:
            for key in filt:
                if key not in FILT_KEYS:
                    raise ValueError(f"Unknown filt key '{key}'. Valid keys: {list(FILT_KEYS)}")

        def passes_filt(halo, galaxy, filt):
            if filt is None:
                return True
            for key, value in filt.items():
                if not FILT_KEYS[key](halo, galaxy, value):
                    return False
            return True

        palette = _Get_Colours(4, self.dark_mode)[1:]
        status_colours = {1: palette[0], 0: palette[1], -1: palette[2]}
        status_labels = {1: 'Visible', 0: 'Partial', -1: 'Invisible'}
        statuses = (status_only,) if status_only is not None else (1, 0, -1)

        need_compute = (weight == 'compute' or parameter.lower() == 'compute' or (parameter2 is not None and parameter2.lower() == 'compute'))
        sweeping = sweep_param is not None

        if sweeping:
            cmap = plt.get_cmap(sweep_cmap)
            n_sweep = len(sweep_values) - 1
            sweep_colours = [cmap(i / max(n_sweep - 1, 1)) for i in range(n_sweep)]
            sweep_labels = [f'{sweep_param} [{sweep_values[i]:.3g}, {sweep_values[i+1]:.3g})'
                            for i in range(n_sweep)]

        # ---- helpers -------------------------------------------------------

        def halo_status(halo):
            galaxies = halo['ObservedGalaxies'] or []
            vis = [g['Visible'] for g in galaxies]
            if len(vis) == 0:
                return -1
            if all(v == -1 for v in vis):
                return -1
            if any(v == -1 for v in vis) or max(vis) == 0:
                return 0
            return 1

        def get_param(halo, galaxy, key):
            key_l = key.lower()
            halo_map = {'redshift': halo['Redshift'],
                        'totalmass': halo['TotalMass'],
                        'gasmass': halo['GasMass'],
                        'stellarmass': halo['StellarMass'],
                        'numstars': halo['NumStars'],
                        'impactparam': halo['ImpactParam']/halo['Radius'] if halo['ImpactParam'] is not None else None,
                        'radius': halo['Radius']}
            
            if need_compute:
                halo_map['compute'] = halo['Compute']

            galaxy_map = {'stellarmass': galaxy['StellarMass'] if galaxy else None,
                        'masslightratio': galaxy['MassLightRatio'] if galaxy else None}
            if galaxy is not None and key_l in galaxy_map and galaxy_map[key_l] is not None:
                return galaxy_map[key_l]
            if key_l in halo_map:
                return halo_map[key_l]
            raise ValueError(f"Unknown parameter '{key}'")

        def sweep_bin(halo, galaxy):
            """Return the sweep group index for this halo/galaxy, or None if out of range."""
            val = get_param(halo, galaxy, sweep_param)
            idx = np.digitize([val], sweep_values)[0] - 1
            if idx < 0 or idx >= n_sweep:
                return None
            return idx

        def iter_records(sl):
            """Yield (param, param2, status, weight_val, sweep_idx) for each entity in a sightline,
            after applying filt at the halo/galaxy level."""
            halos = sl.halo_info(with_compute=need_compute)
            for halo in halos.values():
                if entity == 'halo':
                    if not passes_filt(halo, None, filt):
                        continue
                    s_idx = sweep_bin(halo, None) if sweeping else None
                    if sweeping and s_idx is None:
                        continue
                    status = halo_status(halo)
                    w = halo.get('Compute', 1.0) if need_compute else 1.0
                    p1 = get_param(halo, None, parameter)
                    p2 = get_param(halo, None, parameter2) if parameter2 else None
                    yield p1, p2, status, w, s_idx
                else:  # galaxy-level
                    for gal in (halo['ObservedGalaxies'] or []):
                        if not passes_filt(halo, gal, filt):
                            continue
                        s_idx = sweep_bin(halo, gal) if sweeping else None
                        if sweeping and s_idx is None:
                            continue
                        status = gal['Visible']
                        p1 = get_param(halo, gal, parameter)
                        p2 = get_param(halo, gal, parameter2) if parameter2 else None
                        yield p1, p2, status, 1.0, s_idx

        records = []
        for sl in tqdm(sightlines, desc=f'Gathering {entity} visibility'):
            records.extend(iter_records(sl))

        with self._style():

            fig, ax = plt.subplots()
            title_suffix = f' ({status_labels[status_only]} only)' if status_only is not None else ''
            ax.set_title(f'{self.sim.name} {entity.capitalize()} Visibility{title_suffix}')
            if xscale == 'log':
                ax.set_xscale('log')
            if yscale == 'log':
                ax.set_yscale('log')

            legend_handles = []

            if len(records) == 0:
                print('No halos/galaxies passed the given filt/sweep_values.')
                plt.show()
                return

            p1_arr = np.array([r[0] for r in records], dtype=float)
            p2_arr = np.array([r[1] for r in records], dtype=float) if parameter2 else None
            status_arr = np.array([r[2] for r in records])
            w_arr = np.array([r[3] for r in records], dtype=float)
            sweep_idx_arr = np.array([r[4] for r in records]) if sweeping else None

            sweep_groups = range(n_sweep) if sweeping else [None]

            for s_idx in sweep_groups:

                if sweeping:
                    group_mask = sweep_idx_arr == s_idx
                    if not np.any(group_mask):
                        print(f'Skipping sweep group {sweep_labels[s_idx]}: 0 entries')
                        continue
                    sweep_color = sweep_colours[s_idx]
                    label = sweep_labels[s_idx]
                else:
                    group_mask = np.ones(len(records), dtype=bool)
                    sweep_color = None
                    label = None

                gp1, gp2 = p1_arr[group_mask], (p2_arr[group_mask] if parameter2 else None)
                gstatus, gw = status_arr[group_mask], w_arr[group_mask]

                # ---- fraction plot ----
                if plottype == 'fraction':
                    edges = np.linspace(np.nanmin(gp1), np.nanmax(gp1), bins + 1)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    bin_idx = np.digitize(gp1, edges) - 1
                    bin_idx = np.clip(bin_idx, 0, bins - 1)

                    for status in statuses:
                        frac = np.full(bins, np.nan)
                        for b in range(bins):
                            in_bin = bin_idx == b
                            total_w = gw[in_bin].sum()
                            if total_w > 0:
                                status_w = gw[in_bin & (gstatus == status)].sum()
                                frac[b] = status_w / total_w
                        line_color = sweep_color if sweeping else status_colours[status]
                        ls = {1: '-', 0: '--', -1: ':'}[status]
                        ax.plot(centers, frac, ls, marker='x', color=line_color)

                    ax.set_xlabel(parameter.capitalize())
                    ylab = 'Fraction of DM' if weight == 'compute' else 'Fraction of Galaxies'
                    ax.set_ylabel(ylab)

                    if not sweeping:
                        for status in statuses:
                            legend_handles.append(Line2D([0], [0], color=status_colours[status],
                                                        linestyle={1: '-', 0: '--', -1: ':'}[status],
                                                        marker='x', label=status_labels[status]))
                    else:
                        legend_handles.append(Line2D([0], [0], color=sweep_color, linestyle='-', label=label))

                # ---- scatter plot ----
                elif plottype == 'scatter':
                    if parameter2 is None:
                        raise ValueError("plottype='scatter' requires parameter2 to be set")
                    for status in reversed(statuses) if status_only is None else statuses:
                        mask = gstatus == status
                        if not np.any(mask):
                            continue
                        color = sweep_color if sweeping else status_colours[status]
                        ax.scatter(gp1[mask], gp2[mask], color=color, s=1, alpha=0.5,
                                label=(status_labels[status] if (not sweeping and status_only is None) else None))
                    ax.set_xlabel(parameter.capitalize())
                    ax.set_ylabel(parameter2.capitalize())
                    if not sweeping and status_only is None:
                        legend_handles = [Line2D([0], [0], marker='o', linestyle='none',
                                                color=status_colours[s], label=status_labels[s])
                                        for s in (-1, 0, 1)]
                    elif sweeping:
                        legend_handles.append(Line2D([0], [0], marker='o', linestyle='none',
                                                    color=sweep_color, label=label))

                # ---- histogram ----
                elif plottype == 'hist':
                    edges = np.linspace(np.nanmin(gp1), np.nanmax(gp1), bins + 1)
                    for status in statuses:
                        mask = gstatus == status
                        line_color = sweep_color if sweeping else status_colours[status]
                        ax.hist(gp1[mask], bins=edges, weights=gw[mask] if need_compute else None,
                                histtype='step', color=line_color,
                                label=(status_labels[status] if (not sweeping and status_only is None) else None),
                                linewidth=1.5)
                    ax.set_xlabel(parameter.capitalize())
                    ax.set_ylabel('DM' if weight == 'compute' else 'Count')
                    if not sweeping and status_only is None:
                        legend_handles = [Line2D([0], [0], color=status_colours[s], label=status_labels[s])
                                        for s in (1, 0, -1)]
                    elif sweeping:
                        legend_handles.append(Line2D([0], [0], color=sweep_color, linestyle='-', label=label))

                else:
                    raise ValueError(f"Unknown plottype '{plottype}'")

            ax.legend(handles=legend_handles if legend_handles else None)
            plt.show()

        



        
    # def distribution(self,sightlines,functype='DM',cutoff=98,bins=100,redshift=None,xlims=None,gif_path=None,environment='Total',data='truth'):

    #     colours = {'Total':_Get_Colours(3,self.dark_mode)[0],'CGM':_Get_Colours(3,self.dark_mode)[1],'IGM':_Get_Colours(3,self.dark_mode)[2]}

    #     max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment=environment,modelled= 'model' in data )
    #     redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
    #     redshifts = [z for z in redshift_input if z <= max_redshift]
    #     if max(redshift_input) > max_redshift:
    #         redshifts.append(max_redshift)

    #     all_vals = []
    #     data_source = []
    #     if 'truth' in data:
    #         vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled=False) for sl in tqdm(sightlines,desc=f'Getting Truth {environment} Compute')])
    #         all_vals.append(vals)
    #         data_source.append('Truth')
    #     if 'model' in data:
    #         vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled=True) for sl in tqdm(sightlines,desc=f'Getting Modelled {environment} Compute')])
    #         all_vals.append(vals)
    #         data_source.append('Model')
        
    #     frames = []
    #     for i in tqdm(range(len(redshifts))):
    #         z = redshifts[i]

    #         with self._style():

    #             fig,ax = plt.subplots()
    #             ax.set_title(f'{self.sim.name} {environment} {functype} Distribution to z = {z:.2g}')
    #             ax.set_xlabel(functype)
    #             ax.set_ylabel('Probability Density')
    #             for j in range(len(all_vals)):
    #                 vs = all_vals[j][:,i]
    #                 maxval = np.percentile(vs, cutoff)
    #                 ax.hist(vs[vs<maxval],bins=bins,density=True,label=data_source[j],color=colours[environment])
                    
    #             ax.legend()

    #             if xlims is not None:
    #                 ax.set_xlim(xlims[0],xlims[1])

    #             if gif_path is not None:
                    

    #                 buf = io.BytesIO()
    #                 fig.savefig(buf, format='png', dpi=100)
    #                 buf.seek(0)
    #                 frames.append(imageio.imread(buf))
    #                 buf.close()
    #                 plt.close(fig)
    #             else:
    #                 plt.show()
        
    #     if gif_path is not None:
    #         # fps = 4
    #         # duration=1000 * 1/fps
    #         imageio.mimsave(f'{gif_path}/{functype}_distribution.gif', frames, duration=5000/len(frames))

    # def cumulutive_stats(self,sightlines,stat='mean',functype='DM',bins=10,redshift=None,yscale='linear',environment='Total',data='truth'):

    #     colours = {'Total':_Get_Colours(3,self.dark_mode)[0],'CGM':_Get_Colours(3,self.dark_mode)[1],'IGM':_Get_Colours(3,self.dark_mode)[2]}
    #     if environment == 'separate':
    #         environments = ['Total','CGM','IGM']
    #         check_env = 'CGM'
    #     else:
    #         environments = [environment]
    #         check_env = environment

    #     truth = 'truth' in data
    #     modelled = 'model' in data

    #     stat_map = {'mean': {'func':np.nanmean, 'name': 'Mean'},
    #                     'std' : {'func':np.nanstd, 'name': 'Standard Deviation'}}

    #     with self._style():

    #         fig,ax = plt.subplots()
    #         ax.set_xlabel('Redshift')
    #         ax.set_ylabel(f'Cumulutive {stat_map[stat]["name"]} {functype}')
    #         ax.set_title(f'{self.sim.name} {stat_map[stat]["name"]} {functype}')
    #         if yscale == 'log':
    #             ax.set_yscale('log')

    #         legend_handles = []
    #         for env in environments:
    #             legend_handles.append(Line2D([0], [0], color=colours[env], marker='x', linestyle='none', label=env))

    #         if truth:            
    #             redshift = np.nanmin([sightlines[0].redshift_reached(self.sim.cosmo, environment=check_env), np.nan if redshift is None else redshift])
    #             redshifts = np.linspace(0,redshift,bins)
                
    #             for env in environments:
    #                 vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,env,modelled=False) for sl in tqdm(sightlines,desc=f'Getting Truth {env} Compute')])

    #                 func = stat_map[stat]['func']

    #                 stats = func(vals,axis=0)

    #                 ax.plot(redshifts,stats,'x-',color=colours[env])

    #             legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white', linestyle='-',  label='Truth'))
            
    #         if modelled:
    #             redshift = np.nanmin([sightlines[0].redshift_reached(self.sim.cosmo, environment=check_env,modelled=True), np.nan if redshift is None else redshift])
    #             redshifts = np.linspace(0,redshift,bins)
                
    #             for env in environments:
    #                 vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,env,modelled=True) for sl in tqdm(sightlines,desc=f'Getting Mdelled {env} Compute')])

    #                 func = stat_map[stat]['func']

    #                 stats = func(vals,axis=0)

    #                 ax.plot(redshifts,stats,'x--',color=colours[env])

    #             legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white', linestyle='--', label='Model'))

    #         ax.legend(handles=legend_handles)

    #         plt.show()

    
    # def halo_partition(self,sightlines,functype='DM',cutoff=98,redshift=None,plottype='hist',gif_path=None,modelled=False):

    #     max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment='IGM',modelled=modelled)
    #     redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
    #     redshifts = [z for z in redshift_input if z <= max_redshift]
    #     if max(redshift_input) > max_redshift:
    #         redshifts.append(max_redshift)

    #     colours = {'CGM':_Get_Colours(3,self.dark_mode)[1],'IGM':_Get_Colours(3,self.dark_mode)[2]}


    #     data_source = 'truth' if not modelled else 'modelled'
    #     dms_cgm = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment='CGM',modelled=modelled) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} CGM compute')])
    #     dms_igm = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment='IGM',modelled=modelled) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} IGM compute')])

    #     order = np.argsort(dms_cgm[:, -1]+dms_igm[:, -1])

    #     maxval = np.percentile(dms_cgm[:, -1]+dms_igm[:, -1], cutoff)

    #     frames = []
    #     for i in tqdm(range(len(redshifts))):
    #         z = redshifts[i]
    #         # dm_tot = dms_tot[:,i]
    #         dm_cgm = dms_cgm[:,i]
    #         dm_igm = dms_igm[:,i]

    #         dm_tot = dm_cgm + dm_igm

    #         with self._style():
    #             if plottype == 'hist':
    #                 x = np.arange(len(dm_tot))

    #                 fig, ax = plt.subplots()
    #                 ax.fill_between(x, 0, dm_igm[order], label='IGM',color=colours['IGM'])
    #                 ax.fill_between(x, dm_igm[order], dm_igm[order] + dm_cgm[order], label='CGM',color=colours['CGM'])

    #                 ax.set_title(f'{self.sim.name} {functype} Partition to z = {z:.2g}')
    #                 ax.set_xlabel('Sightlines')
    #                 ax.set_ylabel(f'{functype}')
    #                 ax.set_ylim(0, maxval)
    #                 ax.legend()

    #             elif plottype == 'scatter':
    #                 frac_cgm = dm_cgm / dm_tot
    #                 frac_igm = dm_igm / dm_tot

    #                 fig, ax = plt.subplots()
    #                 ax.scatter(dm_tot, frac_igm, s=10, alpha=0.6, label='IGM fraction',c=colours['IGM'])
    #                 ax.scatter(dm_tot, frac_cgm, s=10, alpha=0.6, label='CGM fraction',c=colours['CGM'])

    #                 ax.set_title(f'{self.sim.name} {functype} Partition to z = {z:.2g}')
    #                 ax.set_xlabel(f'Total {functype}')
    #                 ax.set_ylabel(f'Fractional {functype} Contribution')
    #                 ax.legend()

    #             if gif_path is not None:

    #                 buf = io.BytesIO()
    #                 fig.savefig(buf, format='png', dpi=100)
    #                 buf.seek(0)
    #                 frames.append(imageio.imread(buf))
    #                 buf.close()
    #                 plt.close(fig)
    #             else:
    #                 plt.show()
        
    #     if gif_path is not None:
    #         imageio.mimsave(f'{gif_path}/{functype}_partition.gif', frames, duration=5000/len(frames))