"""
Configuration Loader
"""
import os
import yaml
import copy

class Config:
    """Configuration management class"""
    
    def __init__(self, config_dir=None):
        """
        Args:
            config_dir: config 디렉토리 경로 (None이면 자동 탐색)
        """
        # Config 디렉토리 찾기
        if config_dir is None:
            # 현재 파일 기준으로 프로젝트 루트 찾기
            current_file = os.path.abspath(__file__)
            src_config_dir = os.path.dirname(current_file)
            src_dir = os.path.dirname(src_config_dir)
            project_root = os.path.dirname(src_dir)
            config_dir = os.path.join(project_root, 'config')
        
        self.config_dir = config_dir
        self.default_config_path = os.path.join(config_dir, 'default.yaml')
        self.sites_dir = os.path.join(config_dir, 'sites')
        
        # 설정 로드
        self._config = self._load_yaml(self.default_config_path)
        
        # Path templates 보존 (동적 site 변경을 위해)
        self._path_templates = copy.deepcopy(self._config.get('paths', {}))
        
        # 초기 변수 치환
        self._resolve_variables()
    
    def _load_yaml(self, filepath):
        """YAML 파일 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _deep_merge(self, base, override):
        """딕셔너리 재귀적 병합"""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _resolve_variables(self):
        """경로 변수 치환 (단계별로 계산)"""
        # 기본값
        base_path = self._config['project']['base_path']
        site_name = self._config['site']['name']
        
        # 1단계: data_root
        data_root = os.path.join(base_path, 'data')
        
        # 2단계: input_root, output_root
        input_root = os.path.join(data_root, 'input', site_name)
        output_root = os.path.join(data_root, 'output', site_name)
        
        # 전체 컨텍스트 (순서대로!)
        context = {
            'base_path': base_path,
            'site_name': site_name,
            'data_root': data_root,
            'input_root': input_root,
            'output_root': output_root,
        }
        
        # Template에서 paths 재구성
        paths = copy.deepcopy(self._path_templates)
        
        def resolve_dict(d, ctx):
            """딕셔너리 재귀적 치환"""
            result = {}
            for k, v in d.items():
                if isinstance(v, str):
                    try:
                        result[k] = v.format(**ctx)
                    except KeyError as e:
                        # 디버깅용
                        print(f"Warning: Missing variable {e} in '{v}'")
                        print(f"Available variables: {list(ctx.keys())}")
                        raise
                elif isinstance(v, dict):
                    result[k] = resolve_dict(v, ctx)
                else:
                    result[k] = v
            return result
        
        self._config['paths'] = resolve_dict(paths, context)
    
    def load_site(self, site_name):
        """
        사이트별 설정 로드
        
        Args:
            site_name: 사이트 이름 (예: 'HC', 'PC')
        """
        site_config_path = os.path.join(self.sites_dir, f'{site_name}.yaml')
        
        if not os.path.exists(site_config_path):
            raise FileNotFoundError(f"Site config not found: {site_config_path}")
        
        # 사이트 설정 로드
        site_config = self._load_yaml(site_config_path)
        
        # 기본 설정과 병합
        self._config = self._deep_merge(self._config, site_config)
        
        # 경로 재계산 (site_name이 바뀌었으므로)
        self._resolve_variables()
        
        print(f"✓ Loaded site config: {site_name}")
    
    # ===== 속성 접근자 =====
    
    @property
    def project(self):
        """프로젝트 설정"""
        return self._config.get('project', {})
    
    @property
    def site(self):
        """사이트 설정"""
        return self._config.get('site', {})
    
    @property
    def paths(self):
        """경로 설정"""
        return self._config.get('paths', {})
    
    @property
    def analysis(self):
        """분석 설정"""
        return self._config.get('analysis', {})
    
    @property
    def interpolation(self):
        """Interpolation 설정"""
        return self._config.get('interpolation', {})
    
    @property
    def physics(self):
        """물리 상수"""
        return self._config.get('physics', {})
    
    @property
    def plotting(self):
        """플롯 설정"""
        return self._config.get('plotting', {})
    
    def get(self, key, default=None):
        """설정 값 가져오기"""
        return self._config.get(key, default)
    
    def __repr__(self):
        site_name = self._config.get('site', {}).get('name', 'Unknown')
        return f"Config(site={site_name})"