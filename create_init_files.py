"""
누락된 __init__.py 파일 자동 생성
"""
from pathlib import Path

def create_init_files():
    """필요한 모든 __init__.py 생성"""
    
    base_path = Path(__file__).parent
    
    # 생성할 __init__.py와 내용
    init_files = {
        'src/__init__.py': '''"""
CRNP Footprint Analysis Package
"""
__version__ = "2.0.0"
''',
        
        'src/config/__init__.py': '''"""
Configuration management module
"""
from .loader import Config

__all__ = ['Config']
''',
        
        'src/core/__init__.py': '''"""
Core analysis modules
"""
''',
        
        'src/plotting/__init__.py': '''"""
Plotting modules
"""
''',
        
        'src/utils/__init__.py': '''"""
Utility modules
"""
''',
    }
    
    print("="*60)
    print("Creating __init__.py files")
    print("="*60)
    
    created = 0
    existed = 0
    
    for file_path, content in init_files.items():
        full_path = base_path / file_path
        
        # 디렉토리가 없으면 생성
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if full_path.exists():
            print(f"  - {file_path} (already exists)")
            existed += 1
        else:
            full_path.write_text(content, encoding='utf-8')
            print(f"  ✓ {file_path} (created)")
            created += 1
    
    print("\n" + "="*60)
    print(f"Summary: {created} created, {existed} already existed")
    print("="*60)
    
    if created > 0:
        print("\n✓ Files created successfully!")
        print("\nNext step: Run test_config.py")
    else:
        print("\n✓ All files already exist")


if __name__ == "__main__":
    create_init_files()