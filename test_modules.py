"""
모듈 테스트 스크립트
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """모든 모듈 import 테스트"""
    print("="*70)
    print("Testing Module Imports")
    print("="*70)
    
    try:
        from src.config import Config
        print("✓ Config imported")
        
        from src.core import CRNPAnalyzer, CRNPPhysics, SWCInterpolator, FootprintCalculator
        print("✓ Core modules imported")
        
        from src.plotting import (
            plot_panel_swc,
            plot_panel_veg,
            plot_panel_footprint_amoeba,
            plot_panel_depth_layers,
            plot_panel_footprint_crosssection,
            save_all_panels
        )
        print("✓ Plotting modules imported")
        
        from src.utils import convex_hull_mask, build_amoeba_boundary_polyline, plot_sector_rays
        print("✓ Utils imported")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Config 테스트"""
    print("\n" + "="*70)
    print("Testing Config System")
    print("="*70)
    
    try:
        from src.config import Config
        
        # Default config
        config = Config()
        print(f"✓ Default config loaded")
        print(f"  Site: {config.site['name']}")
        print(f"  Bulk density: {config.site['bulk_density']}")
        
        # HC config
        config.load_site('HC')
        print(f"\n✓ HC config loaded")
        print(f"  Site: {config.site['name']}")
        print(f"  Bulk density: {config.site['bulk_density']}")
        print(f"  Analysis dates: {config.site.get('analysis_dates', [])}")
        
        # 경로 확인
        print(f"\n✓ Paths:")
        print(f"  Input root: {config.paths['input_root']}")
        print(f"  GEO file: {os.path.basename(config.paths['input']['geo'])}")
        print(f"  SWC file: {os.path.basename(config.paths['input']['swc'])}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_physics():
    """Physics 모듈 테스트"""
    print("\n" + "="*70)
    print("Testing Physics Module")
    print("="*70)
    
    try:
        from src.config import Config
        from src.core import CRNPPhysics
        import numpy as np
        
        config = Config()
        physics = CRNPPhysics(config)
        
        # Wr 테스트
        r = np.array([0, 50, 100, 150])
        Wr = physics.radial_intensity_function(r)
        print(f"✓ Wr at r={r}: {Wr}")
        
        # Wd 테스트
        d = np.array([0, 10, 20, 30])
        theta = 0.20
        Wd = physics.depth_weighting_function(d, theta, bulk_density=1.4)
        print(f"✓ Wd at d={d} (θ={theta}): {Wd}")
        
        # D 계산
        D = physics.calculate_penetration_depth(theta, bulk_density=1.4)
        print(f"✓ Penetration depth D: {D:.2f} cm")
        
        # N(θ) 계산
        N = physics.neutron_moisture_relationship(theta)
        print(f"✓ N(θ={theta}): {N:.2f} cph")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Physics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    success &= test_imports()
    success &= test_config()
    success &= test_physics()
    
    if success:
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ Some tests failed")
        print("="*70)