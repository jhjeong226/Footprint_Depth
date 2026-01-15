"""
Panel 2: 식생 높이 맵
"""
import numpy as np
import matplotlib.pyplot as plt
from .base import setup_base_axes

def plot_panel_veg(results, save_path, config):
    """
    Panel 2: Vegetation height
    
    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
    """
    max_extent = results["max_extent"]
    
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    
    # 식생 높이 맵 (균일)
    height_map = np.full_like(results["swc_map"], results["veg_height"], dtype=float)
    
    im = ax.imshow(
        height_map,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="YlGn",
        vmin=0, vmax=0.5,
        interpolation="bilinear"
    )
    
    # CRNP 위치
    ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2, zorder=6)

    title = f"Vegetation Height | {results['date']}"
    setup_base_axes(ax, max_extent, title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (m)", fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)