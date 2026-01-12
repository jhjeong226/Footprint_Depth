"""
현재 파일 상태 확인
"""
from pathlib import Path

project_root = Path(__file__).parent

print("="*70)
print("File Content Check")
print("="*70)

# 1. loader.py에서 _resolve_variables 메서드 확인
print("\n1. Checking loader.py...")
loader_path = project_root / "src" / "config" / "loader.py"

if loader_path.exists():
    content = loader_path.read_text(encoding='utf-8')
    
    # site_name 동적 읽기 확인
    if "'site_name': self._config['site']['name']" in content:
        print("  ✓ loader.py: Uses dynamic site_name (CORRECT)")
    elif "'site_name': self._config['site']['name']" in content:
        print("  ✓ loader.py: Uses dynamic site_name (CORRECT)")
    else:
        print("  ✗ loader.py: Uses static 'Default' (WRONG)")
        print("\n  Expected line:")
        print("    'site_name': self._config['site']['name']")
        print("\n  Search for 'def _resolve_variables' and check the context dict")
else:
    print("  ✗ loader.py not found!")

# 2. default.yaml에서 depth colors 확인
print("\n2. Checking default.yaml...")
yaml_path = project_root / "config" / "default.yaml"

if yaml_path.exists():
    content = yaml_path.read_text(encoding='utf-8')
    
    if "colors:" in content and "'#1f77b4'" in content:
        print("  ✓ default.yaml: Has depth colors (CORRECT)")
    elif "colors:" in content and "- '#1f77b4'" in content:
        print("  ✓ default.yaml: Has depth colors (CORRECT)")
    else:
        print("  ✗ default.yaml: Missing depth colors (WRONG)")
        print("\n  Expected format:")
        print("    panel_depth:")
        print("      colors:")
        print("        - '#1f77b4'")
        print("        - '#ff7f0e'")
else:
    print("  ✗ default.yaml not found!")

# 3. HC.yaml 확인
print("\n3. Checking HC.yaml...")
hc_yaml_path = project_root / "config" / "sites" / "HC.yaml"

if hc_yaml_path.exists():
    content = hc_yaml_path.read_text(encoding='utf-8')
    
    if 'name: "HC"' in content:
        print("  ✓ HC.yaml: Has correct site name")
    else:
        print("  ✗ HC.yaml: Missing or incorrect site name")
else:
    print("  ✗ HC.yaml not found!")

print("\n" + "="*70)
print("Diagnostic complete")
print("="*70)