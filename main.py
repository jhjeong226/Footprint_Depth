"""
CRNP Footprint Analysis Main Script
"""
import os
import sys
import argparse
from datetime import date, timedelta
import numpy as np

from src.config import Config
from src.core import CRNPAnalyzer
from src.plotting import save_all_panels

def resolve_dates(dates_config):
    """YAML dates 블록에서 분석 날짜 목록 생성.
    analysis(개별 날짜)와 range(기간)를 병합해 정렬된 고유 목록 반환.
    """
    result = set()

    for d in dates_config.get('analysis', []):
        result.add(str(d))

    r = dates_config.get('range', {})
    if r and r.get('start') and r.get('end'):
        start = date.fromisoformat(str(r['start']))
        end   = date.fromisoformat(str(r['end']))
        step  = timedelta(days=int(r.get('step_days', 1)))
        cur = start
        while cur <= end:
            result.add(cur.isoformat())
            cur += step

    return sorted(result)


def print_header():
    """Print analysis header"""
    print("=" * 70)
    print("CRNP FOOTPRINT ANALYSIS (HORIZONTAL + VERTICAL)")
    print("=" * 70)
    print()
    print("Depth Calculation Notes:")
    print("- Theta (SWC): Uses footprint-weighted average as representative value")
    print("- Rho_bd (bulk density): Uses plot-average value (adjust per plot!)")
    print("- r (distance): Calculated at r=0 (CRNP center, most sensitive)")
    print("- D varies slightly with r: D(r=100m) ~= 96% of D(r=0)")
    print("=" * 70)

def print_results_summary(results):
    """Print analysis results summary"""
    print(f"\n{'='*70}")
    print(f"Analyzing {results['date']}...")
    print(f"{'='*70}")

    # Horizontal footprint
    print(f"\n[HORIZONTAL FOOTPRINT]")
    print(f"  - Predicted SWC: {results['predicted_swc']:.4f} cm3/cm3")
    print(f"  - Mean SWC: {results['mean_swc']:.4f} cm3/cm3")
    print(f"  - Valid sensors: {results['n_valid_sensors']}/{len(results['swc_values'])}")

    # 해석적 R86 (max_extent와 독립적)
    if results.get('R86_analytical') and np.isfinite(results['R86_analytical']):
        print(f"  - R86 (analytical): {results['R86_analytical']:.1f} m  [max_extent 독립]")

    # 그리드 기반 R86 (비교용)
    if results.get('R86_radius') and np.isfinite(results['R86_radius']):
        print(f"  - R86 (grid): {results['R86_radius']:.1f} m  [max_extent={results.get('max_extent', '?')}m 범위]")

    rps = [s["rp"] for s in results.get("R86_phi_sectors", [])]
    if len(rps) > 0:
        print(f"  - R86(phi): min={np.min(rps):.1f}, mean={np.mean(rps):.1f}, max={np.max(rps):.1f} m")

    # Vertical footprint
    dp = results.get('depth_profile', {})
    if dp:
        print(f"\n[VERTICAL FOOTPRINT] (at r={dp.get('distance_r', 0):.0f}m)")
        print(f"  - Theta used: {dp.get('theta_used', 0):.4f} (footprint-weighted average)")
        print(f"  - Rho_bd used: {dp.get('bulk_density_used', 0):.2f} g/cm3")
        print(f"  - Penetration depth D: {dp.get('D', 0):.1f} cm")
        print(f"  - D86 (86% depth): {dp.get('D86', 0):.1f} cm")
        print(f"  - Layer contributions:")
        for layer in dp.get('layer_contributions', []):
            print(f"      * {layer['layer']}: {layer['contribution_pct']:.1f}%")

    # Climate data
    if results.get('pressure_hpa') is not None or results.get('abs_humidity') is not None:
        print(f"\n[CLIMATE DATA]")
        if results.get('pressure_hpa') is not None:
            print(f"  - Pressure: {results['pressure_hpa']:.2f} hPa (scale factor sP={results['pressure_scale_sP']:.4f})")
        if results.get('abs_humidity') is not None:
            print(f"  - Absolute humidity: {results['abs_humidity']:.2f} g/m3")

def main():
    """Main analysis workflow"""
    
    print_header()
    
    # [Step 1] Load configuration
    parser = argparse.ArgumentParser(description="CRNP Footprint Analysis")
    parser.add_argument("site", help="Site name (e.g. HC, PC)")
    args = parser.parse_args()

    print("\n[Step 1] Loading configuration...")
    config = Config(args.site)
    
    print(f"[OK] Loaded site config: {config.site['name']}")
    print(f"[OK] Site: {config.site['name']}")
    print(f"[OK] Bulk density: {config.site['bulk_density']} g/cm3")
    print(f"[OK] Input files:")
    print(f"  - GEO: {config.paths['geo_file']}")
    print(f"  - SWC: {config.paths['swc_file']}")
    print(f"  - Pressure: {config.paths['pressure_file']}")
    
    # [Step 2] Initialize analyzer
    print("\n[Step 2] Initializing analyzer...")
    
    geo_path = os.path.join(config.paths['input_root'], config.paths['geo_file'])
    swc_path = os.path.join(config.paths['input_root'], config.paths['swc_file'])
    pressure_path = os.path.join(config.paths['input_root'], config.paths['pressure_file'])
    
    analyzer = CRNPAnalyzer(
        config=config,
        geo_path=geo_path,
        swc_path=swc_path,
        pressure_path=pressure_path
    )
    
    # [Step 3] Analyze dates
    analysis_dates = resolve_dates(config.get('dates', {}))
    if not analysis_dates:
        print("[WARN] No dates specified in config. Add dates.analysis or dates.range to the site YAML.")
        return

    print(f"\n[Step 3] Analysis dates ({len(analysis_dates)}): {analysis_dates}")

    base_plot_dir = config.paths['output_plots']

    for date_str in analysis_dates:
        print(f"\n{'='*70}")
        print(f"Analyzing {date_str}...")
        print(f"{'='*70}\n")

        try:
            results = analyzer.analyze_single_day(
                date_str=date_str,
                base_veg_height=config.analysis.get('base_veg_height', 0.3),
                max_extent=config.analysis.get('max_extent', 150),
                resolution=config.analysis.get('resolution', 10),
                ddeg=config.analysis.get('sector_degrees', 30),
                use_pressure=config.analysis.get('use_pressure', True),
                P0=config.physics['pressure']['P0'],
                alpha_p=config.physics['pressure']['alpha_p'],
                bulk_density=config.site.get('bulk_density', 1.4)
            )

            print_results_summary(results)

            output_dir = os.path.join(base_plot_dir, date_str.replace('-', ''))
            save_all_panels(config, analyzer, results, output_dir, prefix="crnp")
            
        except Exception as e:
            print(f"[ERROR] Error analyzing {date_str}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Analysis completed!")
    print(f"Output directory: {base_plot_dir}/<date>/")
    print(f"Total panels per date: 6")
    print(f"  - Panel 01: SWC distribution")
    print(f"  - Panel 02: Vegetation map")
    print(f"  - Panel 03: Footprint boundary")
    print(f"  - Panel 04: Depth layers")
    print(f"  - Panel 05: Cross-section (Distance×Depth)")
    print(f"  - Panel 06: Practical footprint (R86 vs detectable distance)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()