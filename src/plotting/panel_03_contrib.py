"""
Panel 3: Footprint contribution with R86 boundary
"""
import numpy as np
import matplotlib.pyplot as plt
from .base import setup_base_axes, robust_norm_for_contribution
from ..utils.geometry import build_amoeba_boundary_polyline, plot_sector_rays

def plot_panel_footprint_amoeba(results, save_path, config,
                                arc_step_deg=1.0,
                                show_sector_rays=True,
                                norm_mode="log",
                                boundary_lw=2.0):
    """
    Panel 3: Footprint with directional boundary
    
    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
        arc_step_deg: 호 간격
        show_sector_rays: 섹터 레이 표시 여부
        norm_mode: "log" or "power"
        boundary_lw: 경계선 두께
    """
    max_extent = results["max_extent"]
    sectors = results["R86_phi_sectors"]
    ddeg = results["ddeg"]
    
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    
    # Contribution 맵
    C = results["contribution"]
    norm, eps = robust_norm_for_contribution(C, mode=norm_mode)
    
    im = ax.imshow(
        C + eps,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower",
        cmap="hot",
        norm=norm,
        interpolation="bilinear"
    )
    
    # 섹터 레이
    if show_sector_rays:
        plot_sector_rays(
            ax, max_extent, ddeg,
            linewidth=0.8, alpha=0.18, color="white", zorder=2
        )
    
    # R86 경계선
    bx, by = build_amoeba_boundary_polyline(sectors, arc_step_deg=arc_step_deg)
    if bx.size > 0:
        ax.plot(
            bx, by,
            color="red",
            linewidth=boundary_lw,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6
        )
    
    # CRNP 위치
    ax.plot(0, 0, "w+", markersize=18, markeredgewidth=3, zorder=7)
    
    # 제목
    title = f"CRNP Footprint Contribution\n{results['date']}"
    if results.get("pressure_hpa") is not None:
        title += f"\nP={results['pressure_hpa']:.1f} hPa, sP={results['pressure_scale_sP']:.3f}"
    
    setup_base_axes(ax, max_extent, title)
    ax.grid(True, alpha=0.25, linestyle="--", color="white")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Contribution (normalized)", fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)