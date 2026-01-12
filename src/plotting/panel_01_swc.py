"""
Panel 1: 토양수분 분포 맵
"""
import numpy as np
import matplotlib.pyplot as plt
from .base import setup_base_axes

def plot_panel_swc(analyzer, results, save_path, config):
    """
    Panel 1: SWC distribution
    
    Args:
        analyzer: CRNPAnalyzer 객체
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
    """
    max_extent = results["max_extent"]
    loc = analyzer.get_sensor_locations()
    swc_values = results["swc_values"]
    valid = results["valid_mask"]
    
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    
    # SWC 맵
    Zi = results["swc_map"]
    Zi_masked = np.ma.masked_invalid(Zi)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(alpha=0.0)
    
    im = ax.imshow(
        Zi_masked,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap=cmap,
        vmin=0.15, vmax=0.40,
        interpolation="bilinear"
    )
    
    # 센서 위치
    ax.scatter(
        loc[valid, 0], loc[valid, 1],
        c=swc_values[valid], s=160, marker="o",
        edgecolors="red", linewidths=1.4,
        cmap="Blues", vmin=0.15, vmax=0.40,
        zorder=5
    )
    
    # 결측 센서
    if np.any(~valid):
        ax.scatter(
            loc[~valid, 0], loc[~valid, 1],
            s=160, marker="x", color="gray", linewidths=2,
            zorder=5
        )
    
    # CRNP 위치
    ax.plot(0, 0, "r+", markersize=18, markeredgewidth=3, zorder=6)
    
    # 제목
    title = f"Soil Moisture Distribution\n{results['date']}"
    if config.interpolation.get('edge_control', False):
        outside_fill = config.interpolation.get('outside_fill', 'nan')
        hull_buffer = config.interpolation.get('hull_buffer_m', 0)
        title += f"\n(edge: {outside_fill}, buffer={hull_buffer:.0f}m)"
    
    setup_base_axes(ax, max_extent, title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SWC (cm³/cm³)", fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)