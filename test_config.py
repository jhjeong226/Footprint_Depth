"""
설정 시스템 테스트 (디버그 버전)
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[0]}")

# Import 테스트
try:
    from src.config import Config
    print("✓ Successfully imported Config")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_hc_config_debug():
    """HC 사이트 설정 테스트 (디버그)"""
    print("\n" + "="*70)
    print("Testing HC Site Config (DEBUG MODE)")
    print("="*70)
    
    try:
        # 1. 기본 설정 로드
        config = Config()
        
        print(f"\n[Step 1] Default config loaded")
        print(f"  Site name: {config.site['name']}")
        print(f"  Input root: {config.paths['input_root']}")
        
        # 2. HC 사이트 로드
        print(f"\n[Step 2] Loading HC site...")
        config.load_site('HC')
        
        print(f"\n[Step 3] After HC loaded")
        print(f"  Site name: {config.site['name']}")
        print(f"  Input root: {config.paths['input_root']}")
        print(f"  GEO file: {config.paths['input']['geo']}")
        
        # 3. 파일 존재 확인
        print(f"\n[Step 4] File check:")
        geo_path = Path(config.paths['input']['geo'])
        swc_path = Path(config.paths['input']['swc'])
        pressure_path = Path(config.paths['input']['pressure'])
        
        print(f"  GEO: {geo_path}")
        print(f"    Exists: {geo_path.exists()}")
        
        print(f"  SWC: {swc_path}")
        print(f"    Exists: {swc_path.exists()}")
        
        print(f"  Pressure: {pressure_path}")
        print(f"    Exists: {pressure_path.exists()}")
        
        # 4. Depth colors 확인
        print(f"\n[Step 5] Depth colors check:")
        colors = config.get('plotting.panel_depth.colors')
        print(f"  Type: {type(colors)}")
        print(f"  Value: {colors}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_hc_config_debug()