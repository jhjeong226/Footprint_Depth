"""
Panel 3: Actual observation footprint with cumulative contribution (%) and sensor locations
"""
import numpy as np
import matplotlib.pyplot as plt
from .base import setup_base_axes
from ..utils.geometry import plot_sector_rays, mass_contour_field, draw_mass_contour

def plot_panel_footprint_cumulative(results, save_path, config,
                                   show_sensor_locations=True,
                                   show_sector_rays=False,
                                   boundary_lw=2.5,
                                   mass_level=0.86,
                                   n_levels=10,
                                   cmap_name='Reds_r',
                                   cumulative_min=0.0,
                                   cumulative_max=1.0):
    """
    Panel 3: 실제 관측반경 (누적 기여도 % 축 + 센서 위치)
    
    핵심 개선:
    - Y축: Cumulative contribution (0% ~ 100%)
    - 86% contour = 정확히 86% 선
    - 센서 위치 표시 (유효/무효 구분)
    - 훨씬 직관적!
    
    해석:
    - 0% ~ 30%: 흰색 (거의 영향 없음)
    - 30% ~ 60%: 연한 색 (작은 영향)
    - 60% ~ 86%: 중간 색 (중간 영향)
    - 86% ~ 100%: 진한 색 (86% contour 안쪽, 강한 영향)
    - ⚫ 검은 점: 유효한 토양수분 센서
    - ✖ 회색 X: 결측된 센서
    
    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
        show_sensor_locations: 센서 위치 표시 여부 (기본: True)
        show_sector_rays: 섹터 레이 표시 여부 (기본: False)
        boundary_lw: 경계선 두께
        mass_level: mass contour 레벨 (기본 0.86)
        n_levels: 색상 구간 개수 (기본 10)
        cmap_name: 컬러맵 ('Reds_r', 'Oranges_r', 'YlOrRd_r' 등)
        cumulative_min: 표시할 최소 누적 기여도 (기본 0.0 = 0%)
        cumulative_max: 표시할 최대 누적 기여도 (기본 1.0 = 100%)
    """
    max_extent = results["max_extent"]
    ddeg = results["ddeg"]
    Xi, Yi = results["Xi"], results["Yi"]
    C = results["contribution"].astype(float)

    fig, ax = plt.subplots(figsize=(9, 8))

    # ========== Cumulative contribution field 계산 ==========
    # 각 픽셀의 누적 기여도 (0 ~ 1)
    Q = mass_contour_field(C)
    
    # Percentage로 변환 (0 ~ 100)
    Q_percent = Q * 100.0
    
    # ========== Discrete color levels ==========
    
    # 표시 범위 (%)
    vmin_pct = cumulative_min * 100.0  # 0%
    vmax_pct = cumulative_max * 100.0  # 100%
    
    # Masked array (vmin_pct 이하는 투명/흰색)
    Q_masked = np.ma.masked_where(Q_percent < vmin_pct, Q_percent)
    
    # Discrete levels (%)
    levels = np.linspace(vmin_pct, vmax_pct, n_levels + 1)
    
    # Colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Contourf로 discrete colors
    im = ax.contourf(
        Xi, Yi, Q_masked,
        levels=levels,
        cmap=cmap,
        extend='neither',  # 0~100% 범위로 고정
        alpha=1.0
    )
    
    # 경계선 (선택적)
    ax.contour(
        Xi, Yi, Q_masked,
        levels=levels[::2],  # 2개씩 건너뛰어서 표시
        colors='gray',
        linewidths=0.4,
        alpha=0.3
    )
    
    # ========== 86% 경계선 강조 ==========
    # 86% contour를 특별히 표시
    ax.contour(
        Xi, Yi, Q_percent,
        levels=[mass_level * 100],  # 86%
        colors='red',
        linewidths=boundary_lw,
        linestyles='solid',
        alpha=0.95,
        zorder=7
    )
    
    # ========== 토양수분 센서 위치 표시 ==========
    if show_sensor_locations:
        # analyzer 객체가 results에 있으면 사용, 없으면 건너뛰기
        if "sensor_locations" in results and "valid_mask" in results:
            loc = results["sensor_locations"]
            valid = results["valid_mask"]

            # 유효한 센서 (검은 원)
            if np.any(valid):
                ax.scatter(
                    loc[valid, 0], loc[valid, 1],
                    s=50, marker="o",
                    facecolors='black',
                    edgecolors='white',
                    linewidths=1.0,
                    alpha=0.85,
                    zorder=8,
                    label='Valid sensors'
                )

            # 결측 센서 (회색 X)
            if np.any(~valid):
                ax.scatter(
                    loc[~valid, 0], loc[~valid, 1],
                    s=50, marker="x",
                    color="gray",
                    linewidths=1.5,
                    alpha=0.7,
                    zorder=8,
                    label='Missing sensors'
                )
    
    # ========== 섹터 레이 (선택적) ==========
    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, 
                        linewidth=0.6, alpha=0.15, color="gray", zorder=2)
    
    # CRNP 위치
    ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2, zorder=9,
           label='CRNP station')

    # ========== 제목 (간소화) ==========
    title = f"CRNP Observation Footprint | {results['date']}"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Distance (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Distance (m)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect('equal')
    
    # ========== Legend (센서 위치 표시된 경우) ==========
    if show_sensor_locations and "sensor_locations" in results:
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                 edgecolor='gray', fancybox=True)
    
    # ========== Colorbar (이제 직관적!) ==========
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cumulative Signal Contribution (%)", 
                   fontsize=12, fontweight="bold")
    
    # Colorbar ticks를 %로 명확하게
    tick_levels = levels[::max(1, len(levels)//6)]  # 최대 6개 tick
    cbar.set_ticks(tick_levels)
    cbar.set_ticklabels([f'{v:.0f}%' for v in tick_levels])
    
    # 86% 위치 강조 (선택적)
    if vmin_pct <= mass_level * 100 <= vmax_pct:
        cbar.ax.axhline(mass_level * 100, color='red', linewidth=2, 
                       linestyle='--', alpha=0.7)
        cbar.ax.text(0.5, mass_level * 100, f' {mass_level:.0%}', 
                    va='center', ha='left', fontsize=10, 
                    color='red', fontweight='bold',
                    transform=cbar.ax.get_yaxis_transform())

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Backward compatibility
plot_panel_footprint_amoeba = plot_panel_footprint_cumulative
plot_panel_footprint_clean = plot_panel_footprint_cumulative