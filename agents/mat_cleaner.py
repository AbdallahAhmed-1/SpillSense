# === agents/mat_cleaner.py =============================================
"""
Pre-processing pipeline for hyperspectral cubes.

Redpoint-AI strategy
--------------------
1. Noise Reduction          → wavelet denoise (CPU) or 3×3 median
                              (GPU on M-series / CUDA, or CPU fallback)
2. Spectral Calibration      → band-wise min-max scaling (placeholder)
3. Spatial Registration      → identity (placeholder)
4. Atmospheric Correction    → dark-object subtraction
5. Dimensional Reduction     → PCA retaining configurable variance
"""

from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi                        # CPU median fallback
from sklearn.decomposition import PCA
from state.state_utils import OilGasRCAState


class MATCleaner:
    def __init__(self, variance: float = 0.99):
        self.variance = variance

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        if not state.get("hsi_data"):
            print("🛑 No hyperspectral data found; skipping clean step.")
            return state

        cleaned: dict = {}
        for fname, mat_dict in state["hsi_data"].items():
            print(f"🔍 Starting clean pipeline for {fname}")
            cube = self._extract_cube(mat_dict)
            if cube is None or cube.size == 0:
                print(f"⚠️  {fname} has no cube data; skipping.")
                continue

            cube = self._noise_reduction(cube)
            cube = self._spectral_calibration(cube)
            cube = self._spatial_registration(cube)
            cube = self._atmospheric_correction(cube)
            reduced, pca_model = self._dimensionality_reduction(cube)

            cleaned[fname] = {
                "clean_cube": cube.astype(np.float32),
                "pca_cube": reduced.astype(np.float32),
                "bands_kept": int(pca_model.n_components_),
            }
            print(f"✅ Cleaned {fname} | PCA bands: {pca_model.n_components_}")

        state["hsi_data_clean"] = cleaned
        print("📦 Cleaned hyperspectral data:", list(cleaned))
        return state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _extract_cube(self, mat_dict: dict):
        """Return a 3-D cube in (H,W,B) order or None."""
        for key in ("cube", "data", "img", "map", "hyper", "cube_raw"):
            if key in mat_dict:
                arr = np.asarray(mat_dict[key]).astype(np.float32)
                break
        else:
            return None

        if arr.ndim == 2:                       # (H,W) → (H,W,1)
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3 and arr.shape[0] < 10:
            arr = arr.transpose(1, 2, 0)        # (B,H,W) → (H,W,B)
        return arr if arr.ndim == 3 else None

    # 1. Noise Reduction ------------------------------------------------
    def _noise_reduction(self, cube: np.ndarray) -> np.ndarray:
        pixels = cube.shape[0] * cube.shape[1]

        # ---------- small cubes → CPU wavelet -------------------------
        if pixels < 300_000:
            try:
                from skimage.restoration import denoise_wavelet
                import importlib; importlib.import_module("pywt")
                denoised = np.empty_like(cube)
                for b in range(cube.shape[2]):
                    if b % 20 == 0:
                        print(f"   CPU wavelet band {b}/{cube.shape[2]-1}")
                    denoised[:, :, b] = denoise_wavelet(
                        cube[:, :, b],
                        method="BayesShrink",
                        mode="soft",
                        wavelet_levels=3,
                        rescale_sigma=True,
                    )
                print("🔇 CPU wavelet denoising applied")
                return denoised.astype(np.float32)
            except (ImportError, ModuleNotFoundError):
                print("ℹ️  Wavelet libs missing – fallback to median")

        # ---------- large cubes → GPU median if available -------------
        try:
            import torch
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else None
            )
            if device:
                print(f"⚡ GPU 3×3 median on {device.upper()}")
                t_cube = torch.from_numpy(cube).to(device)
                denoised = torch.empty_like(t_cube)
                pad = torch.nn.ReflectionPad2d(1)
                for b in range(t_cube.shape[2]):
                    band = t_cube[:, :, b:b + 1]               # (H,W,1)
                    patches = pad(band.permute(2, 0, 1)).unfold(1, 3, 1).unfold(2, 3, 1)
                    med = patches.contiguous()\
                                .view(1, band.shape[0], band.shape[1], -1)\
                                .median(dim=-1).values
                    denoised[:, :, b] = med[0]
                return denoised.cpu().numpy().astype(np.float32)
        except ImportError:
            pass  # PyTorch not installed

        # ---------- CPU median fallback --------------------------------
        print("ℹ️  GPU not available – CPU 3×3 median filter used")
        return ndi.median_filter(cube, size=(3, 3, 1)).astype(np.float32)

    # 2. Spectral Calibration (placeholder) -----------------------------
    def _spectral_calibration(self, cube: np.ndarray) -> np.ndarray:
        mn = cube.min(axis=(0, 1), keepdims=True)
        mx = cube.max(axis=(0, 1), keepdims=True)
        rng = np.where(mx - mn == 0, 1, mx - mn)
        return (cube - mn) / rng

    # 3. Spatial Registration (identity placeholder) --------------------
    def _spatial_registration(self, cube: np.ndarray) -> np.ndarray:
        return cube

    # 4. Atmospheric Correction (dark-object subtraction) ---------------
    def _atmospheric_correction(self, cube: np.ndarray) -> np.ndarray:
        dark = cube.min(axis=(0, 1), keepdims=True)
        return cube - dark

    # 5. PCA dimensionality reduction ----------------------------------
    def _dimensionality_reduction(self, cube: np.ndarray):
        h, w, b = cube.shape
        flat = cube.reshape(-1, b)
        pca = PCA(self.variance, svd_solver="full")
        reduced = pca.fit_transform(flat)
        return reduced.reshape(h, w, pca.n_components_), pca
