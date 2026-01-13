"""
Panel 3: Actual observation footprint with discrete colors
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .base import setup_base_axes
from ..utils.geometry import plot_sector_rays, draw_mass_contour

def plot_panel_footprint_clean(results, save_path, config,
                               show_sector_rays=False,
                               boundary_lw=2.5,
                               mass_level=0.86,
                               n_levels=20,
                               cmap_name='Reds_r',
                               bg_threshold=0.002):
    """
    Panel 3: 실제 관측반경 (깔끔한 버전)
    
    특징:
    - 실제 86% 관측반경만 표시 (Red contour)
    - 영향 없는 곳: 흰색 배경
    - Discrete color levels (n_levels)
    - Contribution 큰 곳이 진하게
    
    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
        show_sector_rays: 섹터 레이 표시 여부 (기본: False)
        boundary_lw: 경계선 두께
        mass_level: mass contour 레벨 (기본 0.86)
        n_levels: 색상 구간 개수 (기본 10)
        cmap_name: 컬러맵 ('Reds', 'Oranges', 'YlOrRd', 'RdPu' 등)
        bg_threshold: 배경 임계값 (이보다 작으면 흰색)
    """
    max_extent = results["max_extent"]
    ddeg = results["ddeg"]
    Xi, Yi = results["Xi"], results["Yi"]
    C = results["contribution"].astype(float)

    fig, ax = plt.subplots(figsize=(9, 8))

    # ========== Discrete color levels ==========
    
    # Contribution 값 범위
    C_valid = C[np.isfinite(C) & (C > bg_threshold)]
    if C_valid.size > 0:
        vmin = float(np.percentile(C_valid, 1.0))
        vmax = float(np.percentile(C_valid, 99.5))
    else:
        vmin, vmax = 0.0001, 0.01
    
    vmin = max(vmin, bg_threshold)
    
    # Masked array (threshold 이하는 투명)
    C_masked = np.ma.masked_where(C < bg_threshold, C)
    
    # Discrete levels
    levels = np.linspace(vmin, vmax, n_levels + 1)
    
    # Colormap (흰색 → 진한 색)
    cmap = plt.get_cmap(cmap_name)
    
    # Contourf로 discrete colors
    im = ax.contourf(
        Xi, Yi, C_masked,
        levels=levels,
        cmap=cmap,
        extend='max',  # vmax 넘는 값은 가장 진한 색
        alpha=1.0
    )
    
    # 경계선 (선택적)
    ax.contour(
        Xi, Yi, C_masked,
        levels=levels[::2],  # 2개씩 건너뛰어서 표시
        colors='gray',
        linewidths=0.4,
        alpha=0.3
    )
    
    # ========== 섹터 레이 (선택적) ==========
    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, 
                        linewidth=0.6, alpha=0.15, color="gray", zorder=2)
    
    # ========== 실제 86% 관측반경 (Red contour) ==========
    draw_mass_contour(ax, Xi, Yi, C, level=mass_level, 
                     color="red", lw=boundary_lw, alpha=0.95, zorder=7)
    
    # CRNP 위치
    ax.plot(0, 0, "r+", markersize=20, markeredgewidth=3.5, zorder=8)
    
    # ========== 제목 ==========
    title = f"CRNP Actual Observation Footprint\n{results['date']}"
    if results.get("pressure_hpa") is not None:
        title += f"\nP={results['pressure_hpa']:.1f} hPa, sP={results['pressure_scale_sP']:.3f}"
    
    title += f"\n\n🔴 Red contour: {mass_level:.0%} signal contribution boundary"
    
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Distance (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance (m)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect('equal')
    
    # ========== Colorbar ==========
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Signal Contribution (normalized)", fontsize=12, fontweight="bold")
    
    # Colorbar ticks를 명확하게
    tick_levels = levels[::max(1, len(levels)//6)]  # 최대 6개 tick
    cbar.set_ticks(tick_levels)
    cbar.set_ticklabels([f'{v:.4f}' for v in tick_levels])

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Backward compatibility
plot_panel_footprint_amoeba = plot_panel_footprint_clean