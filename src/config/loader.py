"""
Configuration loader with path variable substitution
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import copy

class Config:
    """YAML 기반 설정 관리"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 파일 경로. None이면 default.yaml 사용
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_yaml(self.config_path)
        
        # 원본 경로 템플릿 백업 (중요!)
        self._path_templates = copy.deepcopy(self._config.get('paths', {}))
        
        # 변수 치환
        self._resolve_variables()
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """YAML 파일 로드"""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _resolve_variables(self):
        """
        경로 변수 치환 (원본 템플릿에서 다시 생성)
        """
        # 변수 컨텍스트 (동적 site_name!)
        context = {
            'base_path': self._config['project']['base_path'],
            'site_name': self._config['site']['name']
        }
        
        # 원본 템플릿에서 다시 치환 (핵심!)
        paths = copy.deepcopy(self._path_templates)
        
        # 1단계: 기본 경로들
        for key in ['data_root', 'input_root', 'output_root']:
            if key in paths:
                paths[key] = self._substitute(paths[key], context)
                context[key] = paths[key]
        
        # 2단계: input 경로들
        if 'input' in paths:
            for key, value in paths['input'].items():
                paths['input'][key] = self._substitute(value, context)
        
        # 3단계: output 경로들
        if 'output' in paths:
            for key, value in paths['output'].items():
                paths['output'][key] = self._substitute(value, context)
        
        # 치환된 경로로 업데이트
        self._config['paths'] = paths
    
    def _substitute(self, template: str, context: Dict[str, str]) -> str:
        """
        템플릿 문자열에서 변수 치환
        """
        if not isinstance(template, str):
            return template
        
        result = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        
        return result
    
    def load_site(self, site_name: str):
        """
        사이트별 설정 로드 및 병합
        
        Args:
            site_name: 'HC' or 'PC'
        """
        site_path = self.config_path.parent / "sites" / f"{site_name}.yaml"
        
        if not site_path.exists():
            raise FileNotFoundError(f"Site config not found: {site_path}")
        
        site_config = self._load_yaml(site_path)
        
        # site_name 강제 업데이트
        if 'site' not in site_config:
            site_config['site'] = {}
        site_config['site']['name'] = site_name
        
        # 병합
        self._merge_config(site_config)
        
        # 변수 재치환 (원본 템플릿에서 다시!)
        self._resolve_variables()
        
        print(f"✓ Loaded site config: {site_name}")
    
    def _merge_config(self, override: Dict[str, Any]):
        """설정 딕셔너리 병합 (override가 우선)"""
        def merge_dict(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, override)
    
    def get(self, key_path: str, default=None) -> Any:
        """
        점 표기법으로 설정값 가져오기
        
        Examples:
            >>> config.get('analysis.max_extent')
            150
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, {})
            else:
                return default
        
        if value == {} or value is None:
            return default
        
        return value
    
    def ensure_output_dirs(self):
        """출력 디렉토리 생성"""
        output_paths = self._config['paths']['output']
        
        for key, path in output_paths.items():
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {path}")
    
    # 편의 프로퍼티들
    @property
    def paths(self) -> Dict[str, Any]:
        return self._config.get('paths', {})
    
    @property
    def analysis(self) -> Dict[str, Any]:
        return self._config.get('analysis', {})
    
    @property
    def interpolation(self) -> Dict[str, Any]:
        return self._config.get('interpolation', {})
    
    @property
    def physics(self) -> Dict[str, Any]:
        return self._config.get('physics', {})
    
    @property
    def plotting(self) -> Dict[str, Any]:
        return self._config.get('plotting', {})
    
    @property
    def site(self) -> Dict[str, Any]:
        return self._config.get('site', {})
    
    @property
    def output(self) -> Dict[str, Any]:
        return self._config.get('output', {})
    
    def __repr__(self):
        return f"Config(site={self.site.get('name', 'Unknown')})"