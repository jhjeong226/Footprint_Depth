"""
CRNP Footprint 계산
"""
import numpy as np

class FootprintCalculator:
    """Footprint 계산 클래스"""
    
    def __init__(self, physics):
        """
        Args:
            physics: CRNPPhysics 객체
        """
        self.physics = physics
    
    def calculate_R86_radial(self, Xi, Yi, contribution, target=0.86):
        """
        방사형 R86 계산
        
        Args:
            Xi, Yi: 그리드 좌표
            contribution: 기여도 맵
            target: 목표 누적 기여도 (0.86 = 86%)
        
        Returns:
            R86_radius: R86 반경
            R86_cum: 실제 누적 기여도
        """
        R = np.sqrt(Xi**2 + Yi**2).ravel()
        C = contribution.ravel()
        
        order = np.argsort(R)
        R_sorted = R[order]
        C_sorted = C[order]
        
        cs = np.cumsum(C_sorted)
        idx = np.searchsorted(cs, target)
        
        if idx >= len(R_sorted):
            return np.nan, np.nan
        
        return float(R_sorted[idx]), float(cs[idx])
    
    def directional_Rp_by_sector(self, Xi, Yi, contribution, 
                                 p=0.86, ddeg=30, min_sector_mass=1e-4):
        """
        방향별 R86(φ) 계산
        
        Args:
            Xi, Yi: 그리드 좌표
            contribution: 기여도 맵
            p: 목표 누적 비율
            ddeg: 섹터 각도 (도)
            min_sector_mass: 최소 섹터 질량
        
        Returns:
            sectors: [{'phi0': 0, 'phi1': 30, 'rp': 120}, ...]
        """
        x = Xi.ravel()
        y = Yi.ravel()
        c = contribution.ravel()
        
        r = np.sqrt(x*x + y*y)
        phi = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        
        sectors = []
        
        for phi0 in np.arange(0, 360, ddeg):
            phi1 = phi0 + ddeg
            m = (phi >= phi0) & (phi < phi1)
            
            if not np.any(m):
                continue
            
            rr = r[m]
            cc = c[m]
            sector_mass = cc.sum()
            
            if sector_mass < min_sector_mass:
                continue
            
            # 누적 분포
            order = np.argsort(rr)
            rr = rr[order]
            cc = cc[order]
            cs = np.cumsum(cc)
            
            target = p * sector_mass
            k = np.searchsorted(cs, target)
            k = min(k, len(rr) - 1)
            
            rp = float(rr[k])
            sectors.append({
                "phi0": float(phi0),
                "phi1": float(phi1),
                "rp": rp
            })
        
        return sectors
    
    def compute_depth_profile(self, theta, bulk_density, r=0, 
                             max_depth=100, n_points=100):
        """
        깊이 프로파일 계산
        
        Args:
            theta: 토양 수분
            bulk_density: 토양 밀도
            r: CRNP로부터 거리 (m)
            max_depth: 최대 깊이 (cm)
            n_points: 계산 포인트 수
        
        Returns:
            depth_profile: 딕셔너리
        """
        depths = np.linspace(0, max_depth, n_points)
        
        # D 계산
        D = self.physics.calculate_penetration_depth(theta, bulk_density, r=r)
        
        # 가중치 계산
        weights = self.physics.depth_weighting_function(depths, theta, bulk_density, r)
        weights_norm = weights / np.trapz(weights, depths)
        
        # 누적 분포
        cumulative = np.array([
            np.trapz(weights_norm[:i+1], depths[:i+1])
            for i in range(len(depths))
        ])
        
        # D86 (86% 깊이)
        idx86 = np.searchsorted(cumulative, 0.86)
        idx86 = min(idx86, len(depths) - 1)
        D86 = float(depths[idx86])
        
        # 레이어별 기여도
        layers = [(0, 10), (10, 20), (20, 40), (40, 100)]
        layer_contrib = []
        
        for d_start, d_end in layers:
            idx_layer = (depths >= d_start) & (depths <= d_end)
            if np.any(idx_layer):
                contrib = np.trapz(weights_norm[idx_layer], depths[idx_layer])
                layer_contrib.append({
                    'layer': f'{d_start}-{d_end}cm',
                    'depth_start': d_start,
                    'depth_end': d_end,
                    'contribution': float(contrib),
                    'contribution_pct': float(contrib * 100)
                })
        
        return {
            'depths': depths,
            'weights': weights_norm,
            'cumulative': cumulative,
            'D': float(D),
            'D86': D86,
            'layer_contributions': layer_contrib,
            'theta_used': float(theta),
            'bulk_density_used': float(bulk_density),
            'distance_r': float(r)
        }