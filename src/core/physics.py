"""
CRNP 물리 방정식
"""
import numpy as np

class CRNPPhysics:
    """CRNP 물리 계산 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: Config 객체
        """
        self.config = config
        self.physics = config.physics
    
    def radial_intensity_function(self, r, h=5.0, theta=0.20):
        """
        수평 가중 함수 Wr(r)
        
        Based on: Schrön et al. (2017)
        
        Args:
            r: 거리 (m) - 배열 가능
            h: 공기 습도 (g/m³)
            theta: 토양 수분 (m³/m³)
        
        Returns:
            Wr: 수평 가중값
        """
        params = self.physics['radial_weighting']
        a1 = params['a1']
        b1 = params['b1']
        a2 = params['a2']
        b2 = params['b2']
        c = params['c']
        
        Wr = (a1 * np.exp(-r / b1) + a2 * np.exp(-r / b2)) * (1 - np.exp(-c * r))
        
        return Wr
    
    def depth_weighting_function(self, d, theta, bulk_density=1.4, r=0):
        """
        깊이 가중 함수 Wd(d)
        
        Based on: Wang et al. (2023), Equation (6)
        
        Args:
            d: 깊이 (cm) - 배열 가능
            theta: 토양 수분 (m³/m³)
            bulk_density: 토양 밀도 (g/cm³)
            r: CRNP로부터 거리 (m)
        
        Returns:
            Wd: 깊이 가중값
        """
        # Penetration depth 계산
        D = self.calculate_penetration_depth(theta, bulk_density, r)
        
        # Wd = exp(-2*d/D)
        Wd = np.exp(-2.0 * d / D)
        
        return Wd
    
    def calculate_penetration_depth(self, theta, bulk_density=1.4, r=0):
        """
        침투 깊이 D 계산
        
        Based on: Wang et al. (2023), Equation (6)
        
        Args:
            theta: 토양 수분 (m³/m³)
            bulk_density: 토양 밀도 (g/cm³)
            r: CRNP로부터 거리 (m)
        
        Returns:
            D: 침투 깊이 (cm)
        """
        params = self.physics['depth_weighting']
        p0 = params['p0']
        p1 = params['p1']
        p2 = params['p2']
        p3 = params['p3']
        p4 = params['p4']
        
        theta = np.asarray(theta)
        denominator = p4 + theta
        denominator = np.where(denominator == 0, 1e-10, denominator)
        
        term = (p3 + theta) / denominator
        D = (1.0 / bulk_density) * (p0 + p1 * (p2 + np.exp(-r / 100.0) * term))
        
        return float(D) if np.isscalar(theta) else D
    
    def neutron_moisture_relationship(self, theta, h=5.0, N0=1950):
        """
        중성자-토양수분 관계 N(θ)
        
        Based on: Köhli et al. (2021)
        
        Args:
            theta: 토양 수분 (m³/m³)
            h: 공기 습도 (g/m³)
            N0: 건조 토양 중성자 플럭스
        
        Returns:
            N: 중성자 카운트 (cph)
        """
        params = self.physics['neutron_moisture']
        p1 = params['p1']
        p2 = params['p2']
        p3 = params['p3']
        p4 = params['p4']
        p5 = params['p5']
        p6 = params['p6']
        p7 = params['p7']
        p8 = params['p8']
        
        N = N0 * (p1 + p2 * theta) / (p1 + theta) * \
            (p3 + p4 * h + p5 * h**2 + np.exp(-p6 * theta) * (p7 + p8 * h))
        
        return N
    
    def inverse_neutron_moisture(self, N, h=5.0, N0=1950, bulk_density=1.4):
        """
        역변환: N → θ
        
        Based on: Desilets et al. (2010)
        
        Args:
            N: 중성자 카운트
            h: 공기 습도
            N0: 건조 토양 플럭스
            bulk_density: 토양 밀도
        
        Returns:
            theta: 토양 수분 (m³/m³)
        """
        a0 = 0.0808
        a1 = 0.372
        a2 = 0.115
        
        theta = (a0 / (N / N0 - a1) - a2) * bulk_density
        
        return theta
    
    def vegetation_correction_factor(self, theta, veg_height):
        """
        식생 보정 계수
        
        Based on: Schrön et al. (2017)
        
        Args:
            theta: 토양 수분
            veg_height: 식생 높이 (m)
        
        Returns:
            fveg: 식생 보정 계수
        """
        params = self.physics['vegetation']
        c1 = params['c1']
        c2 = params['c2']
        c3 = params['c3']
        
        fveg = 1 - c1 * (1 - np.exp(-c2 * veg_height)) * (1 + np.exp(-c3 * theta))
        
        return fveg