import numpy as np
import matplotlib.pyplot as plt

plt.rc('text',family='serif')

from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from contextlib import contextmanager

from tqdm import tqdm
import imageio
import io
from ._utils import _Get_Colours

class VisualSim():

    def __init__(self,sim,dark_mode=False):

        self.sim = sim

        self.fig = None
        self.ax = None

        self.dark_mode = dark_mode

    @contextmanager
    def _style(self):
        style = 'dark_background' if self.dark_mode else 'default'
        with plt.style.context(style):
            yield

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
            cmap = _Get_Colours(n_subsightlines)
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

        colours = _Get_Colours(n_sightlines)

        self.plot3d()
        for ii,idx in enumerate(idx):
            self.plot_sightline(sightlines[idx],points,n_subsightlines,colour=colours[ii],halos=halos) 



    # ------------- Data results visualisation ------------- #

    def fullsky_image(self,sightlines,functype='DM',cmap=None,colour='dodgerblue',cutoff=99.5,redshift=None,gif_path=None,environment='Total',modelled=False):

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
        vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} {environment} compute')])
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
                hp.mollview(m, unit=functype, title=f'...', 
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


    
    def distribution(self,sightlines,functype='DM',cutoff=98,bins=100,redshift=None,xlims=None,gif_path=None,environment='Total',data='truth'):


        max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment=environment,modelled= 'model' in data )
        redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
        redshifts = [z for z in redshift_input if z <= max_redshift]
        if max(redshift_input) > max_redshift:
            redshifts.append(max_redshift)

        all_vals = []
        data_source = []
        if 'truth' in data:
            vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled=False) for sl in tqdm(sightlines,desc=f'Getting Truth {environment} Compute')])
            all_vals.append(vals)
            data_source.append('Truth')
        if 'model' in data:
            vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment,modelled=True) for sl in tqdm(sightlines,desc=f'Getting Modelled {environment} Compute')])
            all_vals.append(vals)
            data_source.append('Model')
        
        frames = []
        for i in tqdm(range(len(redshifts))):
            z = redshifts[i]

            with self._style():

                fig,ax = plt.subplots()
                ax.set_title(f'{self.sim.name} {functype} Distribution to z = {z:.2g}')
                ax.set_xlabel(functype)
                ax.set_ylabel('Probability Density')
                for j in range(len(all_vals)):
                    vs = all_vals[j][:,i]
                    maxval = np.percentile(vs, cutoff)
                    ax.hist(vs[vs<maxval],bins=bins,density=True,label=data_source[j],color='dodgerblue')
                    
                ax.legend()

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
            # fps = 4
            # duration=1000 * 1/fps
            imageio.mimsave(f'{gif_path}/{functype}_distribution.gif', frames, duration=5000/len(frames))

    def cumulutive_stats(self,sightlines,stat='mean',functype='DM',bins=10,redshift=None,yscale='linear',environment='Total',data='truth'):

        colours = {'Total':'dodgerblue','CGM':'indianred','IGM':'mediumseagreen'}
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

        with self._style():

            fig,ax = plt.subplots()
            ax.set_xlabel('Redshift')
            ax.set_ylabel(f'Cumulutive {stat_map[stat]["name"]} {functype}')
            ax.set_title(f'{self.sim.name} {stat_map[stat]["name"]} {functype}')
            if yscale == 'log':
                ax.set_yscale('log')

            legend_handles = []
            for env in environments:
                legend_handles.append(Line2D([0], [0], color=colours[env], marker='x', linestyle='none', label=env))

            if truth:            
                redshift = np.nanmin([sightlines[0].redshift_reached(self.sim.cosmo, environment=check_env), np.nan if redshift is None else redshift])
                redshifts = np.linspace(0,redshift,bins)
                
                for env in environments:
                    vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,env,modelled=False) for sl in tqdm(sightlines,desc=f'Getting Truth {env} Compute')])

                    func = stat_map[stat]['func']

                    stats = func(vals,axis=0)

                    ax.plot(redshifts,stats,'x-',color=colours[env])

                legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white', linestyle='-',  label='Truth'))
            
            if modelled:
                redshift = np.nanmin([sightlines[0].redshift_reached(self.sim.cosmo, environment=check_env,modelled=True), np.nan if redshift is None else redshift])
                redshifts = np.linspace(0,redshift,bins)
                
                for env in environments:
                    vals = np.array([sl.extract_compute(self.sim.cosmo,redshifts,env,modelled=True) for sl in tqdm(sightlines,desc=f'Getting Mdelled {env} Compute')])

                    func = stat_map[stat]['func']

                    stats = func(vals,axis=0)

                    ax.plot(redshifts,stats,'x--',color=colours[env])

                legend_handles.append(Line2D([0], [0], color='black' if not self.dark_mode else 'white', linestyle='--', label='Model'))

            ax.legend(handles=legend_handles)

    
    def halo_partition(self,sightlines,functype='DM',cutoff=98,redshift=None,plottype='hist',gif_path=None,modelled=False):

        max_redshift = sightlines[0].redshift_reached(self.sim.cosmo, environment='IGM',modelled=modelled)
        redshift_input = np.atleast_1d(redshift if redshift is not None else max_redshift)
        redshifts = [z for z in redshift_input if z <= max_redshift]
        if max(redshift_input) > max_redshift:
            redshifts.append(max_redshift)

        colours = {'CGM':'indianred','IGM':'mediumseagreen'}

        data_source = 'truth' if not modelled else 'modelled'
        dms_cgm = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment='CGM',modelled=modelled) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} CGM compute')])
        dms_igm = np.array([sl.extract_compute(self.sim.cosmo,redshifts,environment='IGM',modelled=modelled) for sl in tqdm(sightlines,desc=f'Getting {data_source.capitalize()} IGM compute')])

        order = np.argsort(dms_cgm[:, -1]+dms_igm[:, -1])

        maxval = np.percentile(dms_cgm[:, -1]+dms_igm[:, -1], cutoff)

        frames = []
        for i in tqdm(range(len(redshifts))):
            z = redshifts[i]
            # dm_tot = dms_tot[:,i]
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