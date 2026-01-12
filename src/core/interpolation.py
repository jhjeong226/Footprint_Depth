"""
토양수분 공간 내삽
"""
import numpy as np
from scipy.interpolate import Rbf, griddata
from ..utils.geometry import convex_hull_mask

class SWCInterpolator:
    """토양수분 내삽 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: Config 객체
        """
        self.config = config
        self.interp_config = config.interpolation
    
    def interpolate(self, sensor_locations, swc_values, valid_mask, 
                   max_extent=150, resolution=10):
        """
        토양수분 공간 내삽
        
        Args:
            sensor_locations: (N, 2) 센서 좌표
            swc_values: (N,) 토양수분 값
            valid_mask: (N,) 유효 데이터 마스크
            max_extent: 최대 범위 (m)
            resolution: 그리드 해상도 (m)
        
        Returns:
            Xi, Yi, Zi: 내삽된 그리드
        """
        # 설정 읽기
        method = self.interp_config.get('method', 'rbf')
        edge_control = self.interp_config.get('edge_control', True)
        hull_buffer_m = self.interp_config.get('hull_buffer_m', 15.0)
        outside_fill = self.interp_config.get('outside_fill', 'nan')
        
        # 유효한 데이터만 선택
        x_valid = sensor_locations[valid_mask, 0]
        y_valid = sensor_locations[valid_mask, 1]
        swc_valid = swc_values[valid_mask]
        
        n_valid = len(swc_valid)
        if n_valid == 0:
            raise ValueError("No valid SWC sensors.")
        
        # 센서가 너무 적으면 nearest로 변경
        if n_valid <= 2 and method == "rbf":
            method = "nearest"
        
        # 그리드 생성
        xi = np.arange(-max_extent, max_extent + resolution, resolution)
        yi = np.arange(-max_extent, max_extent + resolution, resolution)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # 내삽 수행
        if method == "rbf":
            # RBF 설정
            rbf_config = self.interp_config.get('rbf', {})
            function = rbf_config.get('function', 'thin_plate')
            smooth = rbf_config.get('smooth', 0)
            
            rbf = Rbf(x_valid, y_valid, swc_valid, 
                     function=function, smooth=smooth)
            Zi = rbf(Xi, Yi)
        else:
            # Griddata (linear, cubic, nearest)
            Zi = griddata((x_valid, y_valid), swc_valid, (Xi, Yi), method=method)
            
            # NaN 채우기
            if np.isnan(Zi).any():
                Zi2 = griddata((x_valid, y_valid), swc_valid, (Xi, Yi), method="nearest")
                Zi = np.where(np.isnan(Zi), Zi2, Zi)
        
        # Edge control
        if edge_control and (n_valid >= 3):
            xy = np.c_[x_valid, y_valid]
            inside = convex_hull_mask(xy, Xi, Yi, buffer_m=hull_buffer_m)
            
            if outside_fill == "nan":
                Zi = Zi.astype(float)
                Zi[~inside] = np.nan
            else:
                # Nearest neighbor로 채우기
                gx = Xi[..., None]
                gy = Yi[..., None]
                d2 = (gx - x_valid[None, None, :])**2 + (gy - y_valid[None, None, :])**2
                nn = np.argmin(d2, axis=2)
                Zi_nn = swc_valid[nn]
                Zi = np.where(inside, Zi, Zi_nn)
        
        # 값 제한 (선택적)
        clip_min = self.interp_config.get('clip_min')
        clip_max = self.interp_config.get('clip_max')
        
        if clip_min is not None or clip_max is not None:
            Zi = np.where(
                np.isnan(Zi), 
                np.nan, 
                np.clip(Zi, clip_min or 0.0, clip_max or 1.0)
            )
        
        return Xi, Yi, Zi