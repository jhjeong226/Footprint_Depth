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
    
    def calculate_attenuation_length(self, theta=None, bulk_density=1.4, h=5.0):
        """
        중성자 감쇠 길이 계산 (깊이 가중 함수용)

        Note: radial_intensity_function은 Schrön et al. (2023) Equation (2)를
        직접 사용하므로 이 함수는 현재 depth_weighting_function에서만 사용됨.

        Args:
            theta: 토양 수분 (m³/m³) - 배열 가능
            bulk_density: 토양 밀도 (g/cm³)
            h: 공기 습도 (g/m³) - 추후 확장용

        Returns:
            L: 감쇠 길이 (m)
        """
        if theta is None:
            theta = 0.20

        theta_grav = np.asarray(theta) * bulk_density
        alpha = 0.006
        beta = 0.012
        L = 1.0 / (alpha + beta * theta_grav + 1e-10)
        L = np.clip(L, 20.0, 200.0)

        return L

    def radial_intensity_function(self, r, h=5.0, theta=0.20, bulk_density=1.4):
        """
        수평 가중 함수 Wr(r, h, θ) - 토양수분/습도 의존적

        Based on: Schrön et al. (2017, 2023), Köhli et al. (2015)

        기본 형태: W*r = (a1 * exp(-r/b1) + a2 * exp(-r/b2)) * (1 - exp(-c*r))

        토양수분과 습도에 따라 감쇠 스케일 (b1, b2)이 조정됨:
        - 건조 토양: footprint 증가 (b1, b2 증가)
        - 습윤 토양: footprint 감소 (b1, b2 감소)

        R86 문헌값 (Schrön et al. 2023, Table 1):
        - θ=5%:  R86 ≈ 218m
        - θ=20%: R86 ≈ 170m
        - θ=30%: R86 ≈ 137m
        - θ=40%: R86 ≈ 121m

        Args:
            r: 거리 (m) - 배열 가능
            h: 공기 습도 (g/m³)
            theta: 토양 수분 (m³/m³) - 스칼라 값 사용 권장
            bulk_density: 토양 밀도 (g/cm³) - 현재 미사용

        Returns:
            Wr: 수평 가중값
        """
        params = self.physics['radial_weighting']
        a1 = params['a1']      # 30.0 (근거리 가중치)
        b1_ref = params['b1']  # 1.6  (기준 근거리 감쇠 스케일, m)
        a2 = params['a2']      # 1.0  (원거리 가중치)
        b2_ref = params['b2']  # 100.0 (기준 원거리 감쇠 스케일, m)
        c = params['c']        # 3.7  (근거리 억제 계수)

        # 대표 토양수분 값 사용 (배열인 경우 평균)
        if hasattr(theta, '__len__'):
            theta_repr = float(np.nanmean(theta))
        else:
            theta_repr = float(theta)

        # 기준 조건: θ=0.20, h=5 g/m³
        theta_ref = 0.20
        h_ref = 5.0

        # 토양수분/습도에 따른 스케일링 (Köhli et al. 2015 기반)
        # f_scale: 건조하면 >1, 습윤하면 <1
        # 문헌값 (Schrön et al. 2023 Table 1) 맞춤:
        #   θ=5% → R86≈218m, θ=20% → R86≈170m, θ=40% → R86≈121m
        f_theta = 0.55 + 1.10 * np.exp(-3.5 * theta_repr)  # θ 의존성
        f_h = 1.0 - 0.005 * (h - h_ref)                     # h 의존성 (작음)
        f_scale = f_theta * f_h
        f_scale = np.clip(f_scale, 0.50, 1.7)

        # 감쇠 스케일 조정
        b1 = b1_ref * f_scale
        b2 = b2_ref * f_scale

        # Wr 계산
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

    def calculate_analytical_R86(self, theta, bulk_density=1.4, h=5.0,
                                  target=0.86, r_max=500, n_points=1000):
        """
        해석적 R86 계산 (max_extent와 독립적)

        Based on: Schrön et al. (2023), Equation (4)

        0.86 × N(h,θ) = ∫_0^R86 Wr(h,θ) dr

        Wr(r)을 직접 적분하여 누적 기여도가 target에 도달하는 반경 계산

        Args:
            theta: 대표 토양 수분 (m³/m³)
            bulk_density: 토양 밀도 (g/cm³)
            h: 공기 습도 (g/m³)
            target: 목표 누적 기여도 (0.86 = 86%)
            r_max: 적분 최대 반경 (m)
            n_points: 적분 포인트 수

        Returns:
            R86: 해석적 R86 반경 (m)
            total_integral: 전체 적분값 (정규화 확인용)
        """
        # 반경 배열 생성 (0 근처 제외, Wr(0)=0)
        r = np.linspace(0.1, r_max, n_points)
        dr = r[1] - r[0]

        # Wr 계산
        Wr = self.radial_intensity_function(r, h=h, theta=theta, bulk_density=bulk_density)

        # 1D 적분: ∫ Wr dr (Equation 4)
        # 누적 적분 (trapezoidal)
        cumulative = np.cumsum(Wr) * dr
        total = cumulative[-1]

        if total <= 0:
            return np.nan, 0.0

        # 정규화된 누적 분포
        cumulative_norm = cumulative / total

        # target에 도달하는 반경 찾기
        idx = np.searchsorted(cumulative_norm, target)
        if idx >= len(r):
            # R86이 r_max를 초과 - 경고 상황
            return float(r[-1]), float(total)

        # 선형 보간으로 더 정확한 R86 계산
        if idx > 0:
            r0, r1 = r[idx-1], r[idx]
            c0, c1 = cumulative_norm[idx-1], cumulative_norm[idx]
            if c1 != c0:
                R86 = r0 + (r1 - r0) * (target - c0) / (c1 - c0)
            else:
                R86 = r[idx]
        else:
            R86 = r[idx]

        return float(R86), float(total)