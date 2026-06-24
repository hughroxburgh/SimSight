# simsight/_fzboost.py

import os
import json
import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


class FZBoostPredictor:
    """
    Pure numpy/xgboost implementation of FlexZBoost prediction.
    No RAIL or flexcode dependencies required at inference time.
    """

    def __init__(self, model_dir):
        """
        Load model from a directory containing:
        - fzboost_metadata.json
        - fzboost_estimator_0.json ... fzboost_estimator_N.json
        """
        with open(f'{model_dir}/fzboost_metadata.json') as f:
            meta = json.load(f)

        self.best_basis     = np.array(meta['best_basis'])
        self.basis_system   = meta['basis_system']
        self.z_min          = meta['z_min']
        self.z_max          = meta['z_max']
        self.bump_threshold = meta['bump_threshold']
        self.sharpen_alpha  = meta['sharpen_alpha']
        self.n_estimators   = meta['n_estimators']

        estimators = []
        for i in range(self.n_estimators):
            est = XGBRegressor()
            est.load_model(f'{model_dir}/fzboost_estimator_{i}.json')
            estimators.append(est)

        self.models = MultiOutputRegressor(XGBRegressor())
        self.models.estimators_ = estimators

    @classmethod
    def load(cls, path=None):
        """Load a pre-saved FZBoostPredictor from a pickle file."""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'data', 'fzboost_predictor.pkl')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path):
        """Save this predictor to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # ── Basis functions ───────────────────────────────────────────────────────

    @staticmethod
    def _cosine_basis(responses, n_basis):
        n_obs  = responses.shape[0]
        basis  = np.empty((n_obs, n_basis))
        responses = responses.flatten()
        basis[:, 0] = 1.0
        for col in range(1, n_basis):
            basis[:, col] = np.sqrt(2) * np.cos(np.pi * col * responses)
        return basis

    @staticmethod
    def _make_grid(n_grid, z_min, z_max):
        return np.linspace(z_min, z_max, n_grid).reshape((n_grid, 1))

    def _evaluate_basis(self, responses, n_basis):
        if self.basis_system == 'cosine':
            return self._cosine_basis(responses, n_basis)
        raise ValueError(f"Basis system {self.basis_system} not supported")

    # ── Post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_single(density, tol=1e-6, max_iter=500):
        hi   = np.max(density)
        lo   = 0.0
        area = np.mean(np.maximum(density, 0.0))
        if area == 0.0:
            density[:] = 1.0
        elif area < 1:
            density /= area
            density[density < 0.0] = 0.0
            return
        for _ in range(max_iter):
            mid  = (hi + lo) / 2
            area = np.mean(np.maximum(density - mid, 0.0))
            if abs(1.0 - area) <= tol:
                break
            if area < 1.0:
                hi = mid
            else:
                lo = mid
        density -= mid
        density[density < 0.0] = 0.0

    def _normalize(self, cde_estimates):
        if cde_estimates.ndim == 1:
            self._normalize_single(cde_estimates)
        else:
            np.apply_along_axis(self._normalize_single, 1, cde_estimates)

    def _remove_bumps_single(self, density, delta):
        bin_size     = 1.0 / len(density)
        area         = 0.0
        left_idx     = 0
        for right_idx, val in enumerate(density):
            if val <= 0.0:
                if area < delta:
                    density[left_idx:(right_idx + 1)] = 0.0
                left_idx = right_idx + 1
                area     = 0.0
            else:
                area += val * bin_size
        if area < delta:
            density[left_idx:] = 0.0
        self._normalize_single(density)

    def _remove_bumps(self, cde_estimates, delta):
        if cde_estimates.ndim == 1:
            self._remove_bumps_single(cde_estimates, delta)
        else:
            np.apply_along_axis(self._remove_bumps_single, 1, cde_estimates, delta=delta)

    def _sharpen(self, cde_estimates, alpha):
        cde_estimates **= alpha
        self._normalize(cde_estimates)

    # ── Colour data ───────────────────────────────────────────────────────────

    def make_color_data(self, mags, mag_errs,
                        bands     = ['g_gaap1p0Mag', 'r_gaap1p0Mag', 'i_gaap1p0Mag', 'z_gaap1p0Mag'],
                        err_bands = ['g_gaap1p0MagErr', 'r_gaap1p0MagErr', 'i_gaap1p0MagErr', 'z_gaap1p0MagErr'],
                        ref_band  = 'i_gaap1p0Mag'):
        """
        Build colour feature array from magnitude arrays.

        Parameters
        ----------
        mags     : array (N, 4) or dict — griz magnitudes
        mag_errs : array (N, 4) or dict — griz magnitude errors

        Returns
        -------
        color_data : array (N, 7) — [i_mag, g-r, r-i, i-z, σ(g-r), σ(r-i), σ(i-z)]
        """
        if not isinstance(mags, dict):
            mags     = {b: mags[:, i]     for i, b in enumerate(bands)}
            mag_errs = {b: mag_errs[:, i] for i, b in enumerate(err_bands)}

        input_data = mags[ref_band]
        for i in range(len(bands) - 1):
            color    = mags[bands[i]] - mags[bands[i + 1]]
            colorerr = np.sqrt(mag_errs[err_bands[i]]**2 + mag_errs[err_bands[i + 1]]**2)
            input_data = np.vstack((input_data, color, colorerr))

        return input_data.T

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, mags, mag_errs, n_grid=301):
        """
        Predict photo-z PDFs and modes from griz magnitudes and errors.

        Parameters
        ----------
        mags     : array (N, 4) — griz magnitudes
        mag_errs : array (N, 4) — griz magnitude errors
        n_grid   : int — number of redshift grid points

        Returns
        -------
        z_mode : array (N,)        — mode of each PDF
        pdfs   : array (N, n_grid) — full PDFs
        z_grid : array (n_grid,)   — redshift grid
        """
        mags     = np.atleast_2d(mags)
        mag_errs = np.atleast_2d(mag_errs)

        color_data = self.make_color_data(mags, mag_errs)

        z_grid_unit = self._make_grid(n_grid, 0.0, 1.0)
        z_basis     = self._evaluate_basis(z_grid_unit, max(self.best_basis) + 1)
        z_basis     = z_basis[:, self.best_basis]

        coefs = self.models.predict(color_data)[:, self.best_basis]
        cdes  = np.matmul(coefs, z_basis.T)

        self._normalize(cdes)
        if self.bump_threshold is not None:
            self._remove_bumps(cdes, self.bump_threshold)
        if self.sharpen_alpha is not None:
            self._sharpen(cdes, self.sharpen_alpha)

        cdes /= (self.z_max - self.z_min)

        z_grid = self._make_grid(n_grid, self.z_min, self.z_max).flatten()
        z_mode = z_grid[np.argmax(cdes, axis=1)]

        return z_mode, cdes, z_grid