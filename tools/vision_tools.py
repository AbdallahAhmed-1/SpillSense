import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from io import BytesIO
import base64

# ===============================
# 🔧 Utility Functions
# ===============================

def fig_to_b64(fig):
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def normalize_img(img):
    return (img - img.min()) / (img.max() - img.min())

# ===============================
# 📊 Core Analysis Pipeline
# ===============================

def run_full_analysis(img, label_map=None, n_components=3):
    h, w, b = img.shape
    pixels = img.reshape(-1, b)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(pixels)
    pca_img = X_pca.reshape(h, w, n_components)

    # Anomaly Detection
    mean_vec = np.mean(pixels, axis=0)
    cov_inv = np.linalg.pinv(np.cov(pixels, rowvar=False))
    dists = np.array([mahalanobis(p, mean_vec, cov_inv) for p in pixels])
    anomaly_map = dists.reshape(h, w)

    # Spectral stats
    band_std = img.std(axis=(0, 1))
    anomaly_band = int(np.argmax(band_std))

    return {
        "image_shape": img.shape,
        "high_anomaly_band": anomaly_band,
        "band_std": band_std.tolist(),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "anomaly_summary": {
            "mean": float(anomaly_map.mean()),
            "max": float(anomaly_map.max()),
            "min": float(anomaly_map.min()),
            "std": float(anomaly_map.std())
        },
        "band_description": f"Band {anomaly_band} has the highest variance and may indicate material change or contamination.",
        "insight_summary": (
            f"Anomaly map shows high intensity in band {anomaly_band}. "
            f"Mahalanobis distance (mean: {anomaly_map.mean():.2f}, max: {anomaly_map.max():.2f}) "
            "suggests possible surface anomaly or material irregularity."
        ),
        "_cache": {
            "pca_img": pca_img,
            "anomaly_map": anomaly_map,
            "center_pixel": img[h//2, w//2, :],
            "label_map": label_map
        }
    }

# ===============================
# Visualization Tools
# ===============================

def plot_band(img, band):
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, band], cmap='gray')
    ax.set_title(f"Spectral Band {band}")
    return fig_to_b64(fig)

def plot_pca_rgb(pca_img):
    normed = normalize_img(pca_img)
    fig, ax = plt.subplots()
    ax.imshow(normed)
    ax.set_title("PCA RGB Composite")
    return fig_to_b64(fig)

def plot_anomaly_map(anomaly_map):
    fig, ax = plt.subplots()
    ax.imshow(anomaly_map, cmap='hot')
    ax.set_title("Anomaly Map (Mahalanobis Distance)")
    return fig_to_b64(fig)

def plot_pca_variance(pca_ratio):
    fig, ax = plt.subplots()
    ax.plot(pca_ratio, marker='o')
    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance Ratio")
    return fig_to_b64(fig)

def plot_spectral_profile(center_pixel):
    fig, ax = plt.subplots()
    ax.plot(center_pixel)
    ax.set_title("Spectral Profile (Center Pixel)")
    ax.set_xlabel("Band")
    ax.set_ylabel("Intensity")
    return fig_to_b64(fig)

def plot_region_profiles(img):
    h, w, b = img.shape
    region_h = h // 2
    region_w = w // 2
    regions = [
        ("Top Left", img[:region_h, :region_w, :]),
        ("Top Right", img[:region_h, region_w:, :]),
        ("Bottom Left", img[region_h:, :region_w, :]),
        ("Bottom Right", img[region_h:, region_w:, :])
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, region in regions:
        profile = region.reshape(-1, b).mean(axis=0)
        ax.plot(profile, label=name)

    ax.set_title("Average Spectral Profiles by Region")
    ax.set_xlabel("Band")
    ax.set_ylabel("Intensity")
    ax.legend()
    return fig_to_b64(fig)


def plot_all(img, result):
    high_band = result["high_anomaly_band"]
    pca_rgb = result["_cache"]["pca_img"]
    anomaly_map = result["_cache"]["anomaly_map"]
    center_pixel = result["_cache"]["center_pixel"]

    normed_rgb = normalize_img(pca_rgb)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].imshow(img[:, :, high_band], cmap='gray')
    axs[0, 0].set_title(f"Band {high_band}")
    axs[0, 1].imshow(normed_rgb)
    axs[0, 1].set_title("PCA RGB")
    axs[1, 0].imshow(anomaly_map, cmap='hot')
    axs[1, 0].set_title("Anomaly Map")
    axs[1, 1].plot(center_pixel)
    axs[1, 1].set_title("Spectral Profile (Center)")
    axs[1, 1].set_xlabel("Band")
    axs[1, 1].set_ylabel("Intensity")
    return fig_to_b64(fig)


def plot_top_anomaly_profiles(img, anomaly_map, top_k=5):
    h, w, b = img.shape
    flat_img = img.reshape(-1, b)
    flat_anom = anomaly_map.flatten()

    top_idx = np.argsort(flat_anom)[-top_k:]
    top_profiles = flat_img[top_idx]

    fig, ax = plt.subplots()
    for i, profile in enumerate(top_profiles):
        ax.plot(profile, label=f"Pixel {i+1}")
    ax.set_title(f"Top-{top_k} Anomalous Spectral Profiles")
    ax.set_xlabel("Band")
    ax.set_ylabel("Intensity")
    ax.legend()
    return fig_to_b64(fig)


def plot_band_variance(band_std):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(band_std)), band_std)
    ax.set_title("Band-wise Variance")
    ax.set_xlabel("Band")
    ax.set_ylabel("Standard Deviation")
    return fig_to_b64(fig)


def plot_label_overlay(base_img, label_map, alpha=0.4):
    if label_map is None:
        return None

    base_gray = normalize_img(base_img[:, :, 0])
    overlay = np.ma.masked_where(label_map == 0, label_map)

    # Custom color map: up to 5 labels
    cmap = ListedColormap(["red", "green", "blue", "yellow", "purple"])
    fig, ax = plt.subplots()
    ax.imshow(base_gray, cmap='gray')
    im = ax.imshow(overlay, cmap=cmap, alpha=alpha)
    ax.set_title("Label Map Overlay")

    # Create legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f"Label {i+1}",
                   markerfacecolor=cmap(i), markersize=10)
        for i in range(min(np.max(label_map), cmap.N))
    ]
    ax.legend(handles=handles, loc='upper right')
    return fig_to_b64(fig)

