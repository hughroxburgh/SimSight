"""
Halo gas density profiles.

Every profile is described by its unit shape u(x; M200c, z), x = r / R200c, defined so that a halo with gas
fraction f_gas has density

    rho(r) = f_gas * f_b * M200c / (4/3 pi R200c^3) * u(r / R200c)

i.e. f_gas is a free amplitude and u carries the shape. `HaloProfile` turns u into densities, enclosed masses and
shell masses; a new profile only has to implement `u`. Analytic profiles are registered in PROFILES by name.

    MNFWProfile          modified NFW (Mathews & Prochaska 2017), normalised so u integrates to 1 inside R200c
    AverageHaloProfile   simulation-average profile tabulated in (z, M200c, x), built from per-halo measurements

Units follow the simulation: M200c [Msun], radii [ckpc], density [Msun / ckpc^3].
"""
import os
import json
import datetime

import numpy as np
from scipy import integrate

_trapezoid = getattr(np, 'trapezoid', None) or np.trapz   # numpy >= 2 renamed trapz
_X_FLOOR = 1e-5                                          # smallest x evaluated


class HaloProfile:
    """Base class: subclasses implement u(x, M200, z, h)."""

    name = None

    def u(self, x, M200, z, h):
        """Dimensionless unit shape at x = r / R200c (array) for one halo of mass M200 [Msun] at redshift z."""
        raise NotImplementedError

    def density(self, r, M200, R200, z, f_b, h, f_gas=1.0):
        """Gas density [Msun / ckpc^3] at radii r [ckpc] for one halo (M200 [Msun], R200 [ckpc])."""
        rho200 = f_b * M200 / (4.0 / 3.0 * np.pi * R200**3)
        return f_gas * rho200 * self.u(np.asarray(r, dtype=float) / R200, M200, z, h)

    __call__ = density

    def enclosed(self, x, M200, z, h):
        """Enclosed unit mass 3 int_0^x u x'^2 dx' (1 at x = 1 for a profile normalised to R200c)."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        lx = np.linspace(np.log(_X_FLOOR), np.log(max(x.max(), 1.0)), 4000)
        integrand = 3 * self.u(np.exp(lx), M200, z, h) * np.exp(3 * lx)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lx))])
        return np.interp(np.log(np.maximum(x, _X_FLOOR)), lx, cum)

    def shell_fractions(self, x_edges, M200, z, h):
        """Unit mass in each shell between consecutive x_edges."""
        return np.diff(self.enclosed(x_edges, M200, z, h))


# =========================================================================================== analytic profiles

class MNFWProfile(HaloProfile):
    """
    Modified NFW (Mathews & Prochaska 2017): rho ~ 1 / (y^(1-alpha) (y0 + y)^(2+alpha)), y = c200 x, with
    c200 = 4.67 (M200 h / 1e14 Msun)^-0.11, normalised so that u integrates to 1 inside R200c.
    alpha = 0, y0 = 1 is NFW; the default alpha = y0 = 2 empties the centre (rho ~ y as y -> 0).
    """

    name = 'mnfw'

    def __init__(self, alpha=2.0, y0=2.0):
        self.alpha, self.y0 = alpha, y0

    @staticmethod
    def concentration(M200, h):
        return 4.67 * (M200 * h / 1e14) ** (-0.11)

    def u(self, x, M200, z, h):
        a, y0 = self.alpha, self.y0
        c200 = self.concentration(M200, h)
        norm_integral, _ = integrate.quad(lambda y: y ** (1.0 + a) / (y0 + y) ** (2.0 + a), 0.0, c200)
        y = np.maximum(c200 * np.atleast_1d(np.asarray(x, dtype=float)), 1e-10)
        return c200**3 / (3.0 * norm_integral) / (y ** (1.0 - a) * (y0 + y) ** (2.0 + a))

    def __repr__(self):
        return f'MNFWProfile(alpha={self.alpha}, y0={self.y0})'


PROFILES = {'mnfw': MNFWProfile}


# =========================================================================================== simulation average

_KIND_KEYS = {'gas': 'shell_mass', 'nosf': 'shell_mass_nosf', 'e': 'shell_mass_e'}
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_TABLE_SUFFIX = '_avg_halo_profile.npz'
_FORMAT_VERSION = 1
_INNER_SLOPE = (-2.5, 0.0)   # allowed inner power-law slope (> -3 keeps the enclosed mass finite)


def _x_mid(x_edges):
    lo = np.maximum(x_edges[:-1], x_edges[1] / 2)
    return np.sqrt(lo * x_edges[1:])


class AverageHaloProfile(HaloProfile):
    """
    Simulation-average profile, tabulated on (snapshot z, mass bin, x) from per-halo shell measurements
    (simba_dev/25_avg_halo_profile_job.py):

        u_k(x) = < rho_k(x) / (f_b M200c / (4/3 pi R200c^3)) > / < f_gas(<R200c) >

    rho_k is gas kind k ('gas': all gas, 'nosf': non-star-forming, 'e': DM-equivalent -- the gas mass the fully
    ionised conversion in Density_To_DM needs to reproduce the truth free-electron density) and f_gas is the TOTAL
    gas fraction within R200c, so f_gas keeps meaning total gas. For kind 'e', u integrates to <f_e>/<f_gas> < 1
    inside R200c: gas that does not contribute to DM is absorbed into the shape.

    Evaluation interpolates log u linearly in log x (between shell centres), in log M (between the bin medians of
    each snapshot) and in z (between snapshots), clamping beyond the table in M and z. Shells inside the resolution
    radius are replaced by a power law through the innermost resolved shells, u is held constant inside the
    innermost shell centre (x ~ 0.007), and each node curve is renormalised so its integral inside R200c equals the
    measured one.

        prof = AverageHaloProfile.load('SIMBA')        # packaged table, or a path to one
        rho  = prof(r, M200, R200, z, f_b, h)          # h unused; same signature as every HaloProfile
    """

    name = 'average'

    def __init__(self, z, logM, x_edges, u, n_halos, x_res, integral, meta=None):
        self.z        = np.asarray(z, dtype=float)                # (nz,)
        self.logM     = np.asarray(logM, dtype=float)             # (nz, nM) bin medians, NaN where unused
        self.x_edges  = np.asarray(x_edges, dtype=float)          # (nx + 1,)
        self.u_shell  = np.asarray(u, dtype=float)                # (nz, nM, nx) measured shell means
        self.n_halos  = np.asarray(n_halos, dtype=int)            # (nz, nM)
        self.x_res    = np.asarray(x_res, dtype=float)            # (nz, nM) inner edge of the resolved region
        self.integral = np.asarray(integral, dtype=float)         # (nz, nM) sum_{x<1} u dV
        self.meta     = meta or {}
        self.x_mid    = _x_mid(self.x_edges)
        self._lx      = np.log(self.x_mid)
        if np.any(np.diff(self.z) <= 0):
            raise ValueError('table redshifts must be increasing')
        self._build_nodes()

    # ------------------------------------------------------------------ construction / io

    @classmethod
    def from_measurements(cls, files, kind='e', logm_edges=None, min_count=10, min_halos=10):
        """
        Build from per-snapshot measurement files ({SIM}_snap{NNN}_profiles.npz).

        kind      : 'e' (DM-equivalent gas, default), 'nosf' or 'gas'
        min_count : shells whose median particle count (over the bin's halos) is below this are unresolved
        min_halos : mass bins with fewer halos are left empty
        """
        if kind not in _KIND_KEYS:
            raise ValueError(f"kind must be one of {list(_KIND_KEYS)}")
        snaps = []
        for f in files:
            d = np.load(f)
            meta = json.loads(str(d['meta']))
            if _KIND_KEYS[kind] not in d.files:
                raise ValueError(f"{f} has no '{_KIND_KEYS[kind]}' (measured without full DM?)")
            snaps.append((float(d['z']), f, d, meta))
        snaps.sort(key=lambda s: s[0])

        x_edges = snaps[0][2]['x_edges']
        for _, f, d, _ in snaps:
            if not np.allclose(d['x_edges'], x_edges):
                raise ValueError(f'{f} has different radial bins')
        if logm_edges is None:
            logm_edges = np.asarray(snaps[0][3]['logm_edges'])
        i1 = int(np.searchsorted(x_edges, 1.0))
        dvol = np.diff(x_edges**3)
        nz, nm, nx = len(snaps), len(logm_edges) - 1, len(x_edges) - 1

        z = np.array([s[0] for s in snaps])
        logM = np.full((nz, nm), np.nan)
        u = np.full((nz, nm, nx), np.nan)
        n_halos = np.zeros((nz, nm), dtype=int)
        x_res = np.full((nz, nm), np.nan)
        integral = np.full((nz, nm), np.nan)

        for iz, (_, f, d, meta) in enumerate(snaps):
            M = 10**d['logM200']
            norm = (meta['f_b'] * M)[:, None]
            rho = d[_KIND_KEYS[kind]] / norm / dvol
            fgas = d['shell_mass'][:, :i1].sum(1) / norm[:, 0]
            b = np.digitize(d['logM200'], logm_edges) - 1
            for im in range(nm):
                s = b == im
                n_halos[iz, im] = s.sum()
                if s.sum() < min_halos:
                    continue
                u[iz, im] = rho[s].mean(0) / fgas[s].mean()
                logM[iz, im] = np.median(d['logM200'][s])
                integral[iz, im] = np.sum(u[iz, im, :i1] * dvol[:i1])
                low = np.flatnonzero(np.median(d['shell_count'][s], axis=0)[:i1] < min_count)
                x_res[iz, im] = x_edges[low.max() + 1] if len(low) else 0.0

        sources = [{'file': os.path.basename(f), 'sim_name': m.get('sim_name'), 'snap': m.get('snap'),
                    'z': m.get('z'), 'n_per_bin': m.get('n_per_bin'), 'seed': m.get('seed'),
                    'simsight': m.get('simsight')} for _, f, _, m in snaps]
        meta = {'format_version': _FORMAT_VERSION, 'kind': kind, 'logm_edges': list(map(float, logm_edges)),
                'min_count': min_count, 'min_halos': min_halos, 'sources': sources,
                'sim_name': sources[0]['sim_name'], 'f_b': snaps[0][3].get('f_b'),
                'created': datetime.datetime.now().isoformat(timespec='seconds')}
        return cls(z, logM, x_edges, u, n_halos, x_res, integral, meta)

    @classmethod
    def load(cls, name_or_path):
        """Load a table from a path, or by simulation name from the tables packaged with SimSight."""
        if isinstance(name_or_path, cls):
            return name_or_path
        path = name_or_path
        if not os.path.exists(path):
            path = os.path.join(_DATA_DIR, f'{name_or_path}{_TABLE_SUFFIX}')
        if not os.path.exists(path):
            raise FileNotFoundError(f"no profile table '{name_or_path}'; packaged tables: {cls.available()}")
        d = np.load(path)
        meta = json.loads(str(d['meta']))
        meta['loaded_from'] = os.path.abspath(path)
        return cls(d['z'], d['logM'], d['x_edges'], d['u'], d['n_halos'], d['x_res'], d['integral'], meta)

    @staticmethod
    def available():
        if not os.path.isdir(_DATA_DIR):
            return []
        return sorted(f[:-len(_TABLE_SUFFIX)] for f in os.listdir(_DATA_DIR) if f.endswith(_TABLE_SUFFIX))

    def save(self, path):
        meta = {k: v for k, v in self.meta.items() if k != 'loaded_from'}
        np.savez_compressed(path, z=self.z, logM=self.logM, x_edges=self.x_edges, u=self.u_shell,
                            n_halos=self.n_halos, x_res=self.x_res, integral=self.integral,
                            meta=json.dumps(meta))

    # ------------------------------------------------------------------ node curves

    def _build_nodes(self):
        """log u at the shell centres for every (z, mass) node: inner power law + renormalisation."""
        nz, nm, _ = self.u_shell.shape
        self._logu = np.full(self.u_shell.shape, np.nan)
        lx_fine = np.linspace(np.log(_X_FLOOR), 0.0, 4000)
        for iz in range(nz):
            for im in range(nm):
                u = self.u_shell[iz, im]
                if not np.isfinite(self.logM[iz, im]):
                    continue
                ok = (self.x_edges[:-1] >= self.x_res[iz, im]) & (u > 0) & np.isfinite(u)
                if ok.sum() < 3:
                    self.logM[iz, im] = np.nan
                    continue
                logu = np.full(len(u), np.nan)
                logu[ok] = np.log(u[ok])
                i0 = np.flatnonzero(ok)[0]
                # empty / unresolved shells beyond the first resolved one: interpolate across them
                gaps = ~ok & (np.arange(len(u)) > i0)
                if gaps.any():
                    logu[gaps] = np.interp(self._lx[gaps], self._lx[ok], logu[ok])
                # inside the resolution radius: power law through the innermost resolved shells
                fit = slice(i0, i0 + 3)
                slope = np.clip(np.polyfit(self._lx[fit], logu[fit], 1)[0], *_INNER_SLOPE)
                logu[:i0] = logu[i0] + slope * (self._lx[:i0] - self._lx[i0])
                # renormalise so the integral inside R200c matches the measured one
                f = np.exp(self._interp_lx(lx_fine, logu))
                inside = _trapezoid(3 * f * np.exp(3 * lx_fine), lx_fine)
                self._logu[iz, im] = logu + np.log(self.integral[iz, im] / inside)

    def _interp_lx(self, lx, logu):
        """
        Linear in log x through the node curve; constant inside the innermost node (a core, so line integrals
        through the centre stay finite -- the measured profile says nothing below the innermost shell anyway),
        extended linearly beyond the outermost node.
        """
        out = np.interp(lx, self._lx, logu)                 # np.interp holds logu[0] for lx < first node
        hi = lx > self._lx[-1]
        if hi.any():
            s = (logu[-1] - logu[-2]) / (self._lx[-1] - self._lx[-2])
            out[hi] = logu[-1] + s * (lx[hi] - self._lx[-1])
        return out

    def _node_curve(self, logM, z):
        """log u at the shell centres for a halo of mass logM at redshift z (bilinear in z and log M)."""
        zc = np.clip(z, self.z[0], self.z[-1])
        j = int(np.clip(np.searchsorted(self.z, zc) - 1, 0, max(len(self.z) - 2, 0)))
        if len(self.z) == 1:
            zs, wz = [0], [1.0]
        else:
            t = (zc - self.z[j]) / (self.z[j + 1] - self.z[j])
            zs, wz = [j, j + 1], [1 - t, t]
        curve = 0.0
        for iz, w in zip(zs, wz):
            valid = np.flatnonzero(np.isfinite(self.logM[iz]))
            lm, lu = self.logM[iz, valid], self._logu[iz, valid]
            if len(lm) == 1:
                curve = curve + w * lu[0]
                continue
            mc = np.clip(logM, lm[0], lm[-1])
            k = int(np.clip(np.searchsorted(lm, mc) - 1, 0, len(lm) - 2))
            t = (mc - lm[k]) / (lm[k + 1] - lm[k])
            curve = curve + w * ((1 - t) * lu[k] + t * lu[k + 1])
        return curve

    # ------------------------------------------------------------------ evaluation

    def u(self, x, M200, z, h=None):
        x = np.maximum(np.atleast_1d(np.asarray(x, dtype=float)), _X_FLOOR)
        return np.exp(self._interp_lx(np.log(x), self._node_curve(float(np.log10(M200)), float(z))))

    def __repr__(self):
        return (f"AverageHaloProfile({self.meta.get('sim_name')}, kind={self.meta.get('kind')!r}, "
                f"z={np.round(self.z, 2).tolist()}, logM {np.nanmin(self.logM):.1f}-{np.nanmax(self.logM):.1f}, "
                f"{np.isfinite(self.logM).sum()} nodes)")


def get_profile(profile, table=None, **kwargs):
    """
    A HaloProfile from a name: an analytic profile in PROFILES (kwargs are its parameters, e.g. alpha, y0),
    or 'average' with table = a packaged table name, a path, or an AverageHaloProfile. HaloProfile objects pass
    straight through.
    """
    if isinstance(profile, HaloProfile):
        return profile
    if profile == 'average':
        if table is None:
            raise ValueError("profile='average' needs a table (packaged name, path or AverageHaloProfile)")
        return AverageHaloProfile.load(table)
    if profile in PROFILES:
        return PROFILES[profile](**kwargs)
    raise ValueError(f"unknown halo profile {profile!r}; options: {list(PROFILES) + ['average']}")
