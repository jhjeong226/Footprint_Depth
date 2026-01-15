"""
Panel 1: 토양수분 분포 맵 (Fixed version)
"""
import numpy as np
import matplotlib.pyplot as plt
from .base import setup_base_axes

def plot_panel_swc(analyzer, results, save_path, config,
                  use_discrete_colors=None,
                  n_levels=None,
                  cmap_name=None):
    """
    Panel 1: SWC distribution
    
    Args:
        analyzer: CRNPAnalyzer 객체
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
        use_discrete_colors: 구간별 색상 사용 (None이면 기본값)
        n_levels: 색상 구간 개수 (None이면 기본값)
        cmap_name: 컬러맵 이름 (None이면 기본값)
    """
    # ✅ 안전한 기본값 설정
    if use_discrete_colors is None:
        try:
            # Config에서 읽기 시도
            use_discrete_colors = config.get('plotting.panel_swc.use_discrete_colors', True)
            if use_discrete_colors is None:
                use_discrete_colors = True
        except:
            use_discrete_colors = True
    
    if n_levels is None:
        try:
            # Config에서 읽기 시도
            n_levels = config.get('plotting.panel_swc.n_levels', 20)
            if n_levels is None:
                n_levels = 20
        except:
            n_levels = 20
    
    if cmap_name is None:
        try:
            # Config에서 읽기 시도
            cmap_name = config.get('plotting.panel_swc.cmap_name', 'Blues')
            if cmap_name is None:
                cmap_name = 'Blues'
        except:
            cmap_name = 'Blues'
    
    max_extent = results["max_extent"]
    loc = analyzer.get_sensor_locations()
    swc_values = results["swc_values"]
    valid = results["valid_mask"]
    
    fig, ax = plt.subplots(figsize=(7.8, 6.5))
    
    # SWC 맵
    Zi = results["swc_map"]
    Zi_masked = np.ma.masked_invalid(Zi)
    
    vmin, vmax = 0.1, 0.5
    
    # Discrete 또는 continuous 색상 선택
    if use_discrete_colors:
        # 구간별 명확한 색상
        levels = np.linspace(vmin, vmax, int(n_levels) + 1)  # ✅ int() 추가
        
        cmap = plt.get_cmap(cmap_name)
        im = ax.contourf(
            results['Xi'], results['Yi'], Zi_masked,
            levels=levels,
            cmap=cmap,
            extend='both'
        )
        
        # 경계선 (선택적)
        ax.contour(
            results['Xi'], results['Yi'], Zi_masked,
            levels=levels[::2],
            colors='gray',
            linewidths=0.5,
            alpha=0.3
        )
    else:
        # 연속적인 색상 (기존 방식)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(alpha=0.0)
        
        im = ax.imshow(
            Zi_masked,
            extent=[-max_extent, max_extent, -max_extent, max_extent],
            origin="lower", cmap=cmap,
            vmin=vmin, vmax=vmax,
            interpolation="bilinear"
        )
    
    # 센서 위치
    ax.scatter(
        loc[valid, 0], loc[valid, 1],
        c=swc_values[valid], s=60, marker="o",
        edgecolors="red", linewidths=1.0,
        cmap=cmap_name, vmin=vmin, vmax=vmax,
        zorder=5
    )

    # 결측 센서
    if np.any(~valid):
        ax.scatter(
            loc[~valid, 0], loc[~valid, 1],
            s=60, marker="x", color="gray", linewidths=1.5,
            zorder=5
        )

    # CRNP 위치
    ax.plot(0, 0, "r+", markersize=12, markeredgewidth=2, zorder=6)

    # 제목 (간소화)
    title = f"Soil Moisture Distribution | {results['date']}"

    setup_base_axes(ax, max_extent, title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SWC (cm³/cm³)", fontsize=12, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)