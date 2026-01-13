"""
플롯 모듈
"""
from .panel_01_swc import plot_panel_swc
from .panel_02_veg import plot_panel_veg
from .panel_03_cumulative import plot_panel_footprint_cumulative
from .panel_04_depth import plot_panel_depth_layers
from .panel_05_xsection import plot_panel_footprint_crosssection

__all__ = [
    'plot_panel_swc',
    'plot_panel_veg',
    'plot_panel_footprint_amoeba',
    'plot_panel_depth_layers',
    'plot_panel_footprint_crosssection',
]


def save_all_panels(config, analyzer, results, out_dir, prefix="crnp"):
    """
    모든 패널 저장
    
    Args:
        analyzer: CRNPAnalyzer 객체
        results: analyze_single_day 결과
        out_dir: 출력 디렉토리
        config: Config 객체
        prefix: 파일명 접두사
    """
    import os
    
    os.makedirs(out_dir, exist_ok=True)
    d = results["date"].replace("-", "")
    
    print(f"\nSaving panels for {results['date']}...")
    
    # Panel 1: SWC distribution
    plot_panel_swc(
        analyzer, results,
        os.path.join(out_dir, f"{prefix}_{d}_01_swc_distribution.png"),
        config
    )
    
    # Panel 2: Vegetation
    plot_panel_veg(
        results,
        os.path.join(out_dir, f"{prefix}_{d}_02_vegetation_map.png"),
        config
    )
    
    # Panel 3: Footprint contribution
    plot_panel_footprint_cumulative(
        results,
        os.path.join(out_dir, f"{prefix}_{d}_03_footprint.png"),
        config,
        cumulative_min=0.0,   # 0%부터
        cumulative_max=1.0,   # 100%까지
        n_levels=10
    )
    
    # Panel 4: Depth layers
    plot_panel_depth_layers(
        results,
        os.path.join(out_dir, f"{prefix}_{d}_04_depth_layers.png"),
        config
    )
    
    # Panel 5: Cross-section
    plot_panel_footprint_crosssection(
        results,
        os.path.join(out_dir, f"{prefix}_{d}_05_crosssection.png"),
        config,
        bg_threshold=0.0005,
        gamma=0.45
    )
    
    print(f"✓ Saved 5 panels to: {out_dir}")
    print(f"  01: SWC distribution map")
    print(f"  02: Vegetation height map")
    print(f"  03: Footprint contribution with R86(φ) boundary")
    print(f"  04: Vertical depth layer contributions")
    print(f"  05: 2D Cross-section (Distance × Depth)")