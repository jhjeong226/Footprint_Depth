"""
CRNP Footprint Analysis Main Script
"""
import os
from src.config import Config
from src.core.analyzer import CRNPAnalyzer
from src.plotting import save_all_panels

def print_header():
    """Print analysis header"""
    print("=" * 70)
    print("CRNP FOOTPRINT ANALYSIS (HORIZONTAL + VERTICAL)")
    print("=" * 70)
    print()
    print("Depth Calculation Notes:")
    print("- θ (SWC): Uses footprint-weighted average as representative value")
    print("- ρbd (bulk density): Uses plot-average value (adjust per plot!)")
    print("- r (distance): Calculated at r=0 (CRNP center, most sensitive)")
    print("- D varies slightly with r: D(r=100m) ≈ 96% of D(r=0)")
    print("=" * 70)

def print_results_summary(results):
    """Print analysis results summary"""
    print(f"\n{'='*70}")
    print(f"Analyzing {results['date']}...")
    print(f"{'='*70}")
    
    # Horizontal footprint
    print(f"\n🌍 HORIZONTAL FOOTPRINT:")
    print(f"  ✓ Predicted SWC: {results['predicted_swc']:.4f} cm³/cm³")
    print(f"  ✓ Mean SWC: {results['mean_swc']:.4f} cm³/cm³")
    print(f"  ✓ Valid sensors: {results['n_valid_sensors']}/{len(results['swc_values'])}")
    
    if results.get('R86_radius') and np.isfinite(results['R86_radius']):
        print(f"  ✓ R86 (radial): {results['R86_radius']:.1f} m")
    
    rps = [s["rp"] for s in results.get("R86_phi_sectors", [])]
    if len(rps) > 0:
        print(f"  ✓ R86(φ): min={np.min(rps):.1f}, mean={np.mean(rps):.1f}, max={np.max(rps):.1f} m")
    
    # Vertical footprint
    dp = results.get('depth_profile', {})
    if dp:
        print(f"\n📏 VERTICAL FOOTPRINT (at r={dp.get('distance_r', 0):.0f}m):")
        print(f"  ✓ θ used: {dp.get('theta_used', 0):.4f} (footprint-weighted average)")
        print(f"  ✓ ρbd used: {dp.get('bulk_density_used', 0):.2f} g/cm³")
        print(f"  ✓ Penetration depth D: {dp.get('D', 0):.1f} cm")
        print(f"  ✓ D86 (86% depth): {dp.get('D86', 0):.1f} cm")
        print(f"  ✓ Layer contributions:")
        for layer in dp.get('layer_contributions', []):
            print(f"      • {layer['layer']}: {layer['contribution_pct']:.1f}%")
    
    # Pressure
    if results.get('pressure_hpa') is not None:
        print(f"\n🌡️  PRESSURE:")
        print(f"  ✓ {results['pressure_hpa']:.2f} hPa (scale factor sP={results['pressure_scale_sP']:.4f})")

def main():
    """Main analysis workflow"""
    import numpy as np
    
    print_header()
    
    # [Step 1] Load configuration
    print("\n[Step 1] Loading configuration...")
    config = Config()
    config.load_site('HC')  # 또는 'PC'
    
    print(f"✓ Site: {config.site['name']}")
    print(f"✓ Bulk density: {config.site['bulk_density']} g/cm³")
    print(f"✓ Input files:")
    print(f"  - GEO: {config.paths['geo_file']}")
    print(f"  - SWC: {config.paths['swc_file']}")
    print(f"  - Pressure: {config.paths['pressure_file']}")
    
    # [Step 2] Initialize analyzer
    print("\n[Step 2] Initializing analyzer...")
    
    # ===== 경로 구성 =====
    geo_path = os.path.join(
        config.paths['input_root'],
        config.paths['geo_file']
    )
    swc_path = os.path.join(
        config.paths['input_root'],
        config.paths['swc_file']
    )
    pressure_path = os.path.join(
        config.paths['input_root'],
        config.paths['pressure_file']
    )
    
    # ===== Analyzer 초기화 =====
    analyzer = CRNPAnalyzer(
        config=config,
        geo_path=geo_path,
        swc_path=swc_path,
        pressure_path=pressure_path
    )
    
    print(f"✓ Loaded {len(analyzer.sensor_ids)} sensors")
    print(f"✓ Date range: {analyzer.swc_df['Date'].min()} to {analyzer.swc_df['Date'].max()}")
    print(f"✓ CRNP center: ({analyzer.crnp_lat:.6f}, {analyzer.crnp_lon:.6f})")
    if analyzer.pressure_daily is not None:
        print(f"✓ Pressure series loaded: {len(analyzer.pressure_daily)} daily values")
    
    # [Step 3] Analyze dates
    analysis_dates = config.site.get('analysis_dates', [])
    if not analysis_dates:
        analysis_dates = ['2024-08-16', '2024-09-20', '2024-10-26']
    
    print(f"\n[Step 3] Analysis dates: {analysis_dates}")
    
    for date_str in analysis_dates:
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
            
            # Print summary
            print_results_summary(results)
            
            # Save plots
            output_dir = config.paths['output_plots']
            save_all_panels(analyzer, results, output_dir, prefix="crnp")
            
        except Exception as e:
            print(f"✗ Error analyzing {date_str}: {e}")
            import traceback
            traceback.print_exc()
    
    # [Step 4] Summary
    print(f"\n{'='*70}")
    print("Analysis completed!")
    print(f"Output directory: {config.paths['output_plots']}")
    print(f"Total panels per date: 5")
    print(f"  - Panel 01: SWC distribution")
    print(f"  - Panel 02: Vegetation map")
    print(f"  - Panel 03: Footprint boundary")
    print(f"  - Panel 04: Depth layers")
    print(f"  - Panel 05: Cross-section (Distance×Depth)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()