"""
Schrön et al. (2023) Signal Contribution — 미구현 기능 추가 모듈
================================================================

현재 코드는 기여도 맵(Eq.7)·R86 등고선·섹터 R86까지 구현되어 있음.
이 모듈은 다음 세 가지를 추가한다.

1. remote_field_contribution()   — 원거리 밭의 신호 기여도 c2 (Eq.13 + Eq.7)
2. practical_footprint_distance()— Δθ 감지 가능 최대 거리 R (Eq.14 수치해)
   + practical_footprint_approx() — 빠른 근사식 (Eq.15 / Eq.16)
3. rescale_neutron_moisture()    — 불변 구역 damping 제거용 N(θ) 재조정 (Fig.6)

모든 함수는 검증된 CRNPPhysics (physics.py)의
radial_intensity_function(r*), rescale_distance(), neutron_moisture_relationship(),
calculate_analytical_R86() 를 사용한다.
"""
import numpy as np
from scipy.optimize import brentq


class SignalContribution:
    """Schrön 2023 원거리 기여도 / 실용적 footprint 계산."""

    def __init__(self, physics, config=None):
        """
        Parameters
        ----------
        physics : CRNPPhysics
            검증된 physics.py 인스턴스.
        config : Config, optional
            N0 등 사이트 파라미터 접근용.
        """
        self.physics = physics
        self.config = config

    # 내부 헬퍼: N(θ) — config의 N0 사용 -------------------------------- #
    def _N(self, theta, h=5.0):
        if self.config is not None:
            N0 = self.config.physics["neutron_moisture"]["N0"]
        else:
            N0 = 1950.0
        return self.physics.neutron_moisture_relationship(theta, h=h, N0=N0)

    # 내부 헬퍼: 재스케일된 Wr(r) (기압·식생·θ 반영) -------------------- #
    def _Wr_rescaled(self, r, theta, h, pressure_hpa, Hveg, P0=1013.25):
        rstar = self.physics.rescale_distance(
            r, pressure_hpa=pressure_hpa, Hveg=Hveg, theta=theta, P0=P0
        )
        return self.physics.radial_intensity_function(
            rstar, h=h, theta=theta
        )

    # ================================================================== #
    # 1. 원거리 밭 기여도 (Eq. 13 + Eq. 7)                                #
    # ================================================================== #
    def remote_field_weight(self, R, theta_eff, h=5.0,
                            pressure_hpa=1013.25, Hveg=0.0,
                            geometry="cartesian", r_max=600.0, dr=0.1,
                            P0=1013.25):
        """
        원거리 반평면(또는 원형 외부) 밭 A2(x>R)의 공간 가중 w (Eq.13).

        w = (1/π) ∫_R^∞ Wr(h, θ̂) · arccos(R/r) dr     (cartesian, 직선 경계)
        w =        ∫_R^∞ Wr(h, θ̂) dr / ∫_0^∞ Wr dr      (radial, 원형 경계)

        Parameters
        ----------
        R : float
            중심에서 원거리 밭까지의 직교 거리 (m).
        theta_eff : float
            가중에 쓰는 대표(유효) 토양수분 θ̂. 보통 주 밭 θ1.
        geometry : {"cartesian", "radial"}
            직선 경계(전형적 농지) 또는 원형 경계(피벗 관개).

        Returns
        -------
        w : float
            정규화된 공간 가중 (0~1).
        """
        r = np.arange(dr, r_max, dr)
        Wr = self._Wr_rescaled(r, theta_eff, h, pressure_hpa, Hveg, P0)
        total = np.trapezoid(Wr, r)
        if total <= 0:
            return np.nan

        far = r >= R
        if geometry == "cartesian":
            # arccos(R/r): r<R 에서 0, r>=R 에서 호 길이 비율
            arc = np.zeros_like(r)
            arc[far] = np.arccos(np.clip(R / r[far], -1.0, 1.0)) / np.pi
            w = np.trapezoid(Wr * arc, r) / total
        else:  # radial
            w = np.trapezoid(Wr[far], r[far]) / total
        return float(w)

    def remote_field_contribution(self, R, theta_main, theta_remote,
                                  h=5.0, pressure_hpa=1013.25, Hveg=0.0,
                                  geometry="cartesian", P0=1013.25):
        """
        원거리 밭의 신호 기여도 c2 와 유효 평균 토양수분 θ̂ (Eq.13 + Eq.7).

        예시(논문): θ1=5%, θ2=10%, R=57m → c2≈14.7% (Wr), θ̂≈5.1%

        Returns
        -------
        dict : { 'c2', 'c1', 'theta_eff', 'N_total', 'w' }
        """
        # 1차: θ̂ 초기값 = θ1로 가중 계산 (Eq.13의 Wr(h,θ̂))
        theta_eff = theta_main
        for _ in range(5):  # θ̂ 반복 갱신 (논문 권고)
            w = self.remote_field_weight(
                R, theta_eff, h, pressure_hpa, Hveg, geometry, P0=P0
            )
            N1 = self._N(theta_main, h)
            N2 = self._N(theta_remote, h)
            N_total = (1 - w) * N1 + w * N2          # Eq.13
            # 유효 평균 토양수분: N_total을 역변환
            theta_eff_new = self._invert_N(N_total, h)
            if abs(theta_eff_new - theta_eff) < 1e-5:
                theta_eff = theta_eff_new
                break
            theta_eff = theta_eff_new

        c2 = w * N2 / N_total                          # Eq.7
        c1 = (1 - w) * N1 / N_total
        return {
            "c2": float(c2), "c1": float(c1),
            "theta_eff": float(theta_eff),
            "N_total": float(N_total), "w": float(w),
        }

    def _invert_N(self, N_target, h=5.0, lo=0.001, hi=0.60):
        """N(θ)=N_target 를 θ에 대해 수치 역변환 (단조 감소 가정)."""
        f = lambda th: self._N(th, h) - N_target
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            # 범위 밖이면 가까운 경계 반환
            return lo if abs(flo) < abs(fhi) else hi
        return brentq(f, lo, hi, xtol=1e-6)

    # ================================================================== #
    # 2. 실용적 footprint 거리 R (Eq. 14 수치해)                          #
    # ================================================================== #
    def practical_footprint_distance(self, theta_main, dtheta=0.10,
                                     sigma_N=0.01, h=5.0,
                                     pressure_hpa=1013.25, Hveg=0.0,
                                     geometry="cartesian", P0=1013.25,
                                     R_lo=1.0, R_hi=400.0):
        """
        원거리 밭의 Δθ 변화가 감지되는 최대 거리 R (Eq.14).

        조건: |1 - N(θ̂,θ2)/N(θ̂,θ1)| > σN  를 만족하는 최대 R.
        Δθ>0 (wetting) 또는 Δθ<0 (drying) 모두 가능.

        예시(논문 Table 1, σN=1%, Δθ=+10%):
            θ1=5%  → R≈185m ;  θ1=20% → R≈81m ;  θ1=40% → R≈31m

        Returns
        -------
        R : float
            실용적 footprint 거리 (m). 감지 불가 시 0.
        """
        theta1 = theta_main
        theta2 = theta_main + dtheta

        def signal_change(R):
            # N(θ̂,θ1): 균일(원거리도 θ1) — 변화 전
            # N(θ̂,θ2): 원거리만 θ2 — 변화 후
            w = self.remote_field_weight(
                R, theta1, h, pressure_hpa, Hveg, geometry, P0=P0
            )
            N_before = self._N(theta1, h)                      # 균일
            N_after = (1 - w) * self._N(theta1, h) + w * self._N(theta2, h)
            return abs(1.0 - N_after / N_before) - sigma_N     # >0 이면 감지

        # R이 작을수록 변화 큼(감지), 클수록 작아짐 → 부호 바뀌는 지점 탐색
        f_lo = signal_change(R_lo)
        f_hi = signal_change(R_hi)
        if f_lo < 0:
            return 0.0          # 가장 가까워도 감지 불가
        if f_hi > 0:
            return R_hi         # 먼 거리에서도 감지 (상한 반환)
        return float(brentq(signal_change, R_lo, R_hi, xtol=0.1))

    def practical_footprint_approx(self, theta_main, dtheta=0.05,
                                   h=5.0, pressure_hpa=1013.25, Hveg=0.0,
                                   use_R86=True, P0=1013.25):
        """
        실용적 footprint 거리의 빠른 근사식.

        Eq.15 (R86 기반):  R = R86(h,θ,P) · exp(0.31 − 8θ − 5Δθ)
        Eq.16 (R86 모를 때): R ≈ 225 · exp(0.25 − 9θ − 5Δθ)   [m]
          (Δθ=±0.05, σN=1%, 평균 습도·표준기압 가정)
        """
        if use_R86:
            R86 = self.physics.calculate_analytical_R86(
                theta=theta_main, h=h, pressure_hpa=pressure_hpa, Hveg=Hveg, P0=P0
            )
            return float(R86 * np.exp(0.31 - 8.0 * theta_main - 5.0 * dtheta))
        else:
            return float(225.0 * np.exp(0.25 - 9.0 * theta_main - 5.0 * dtheta))

    # ================================================================== #
    # 3. N(θ) 재조정 — 불변 구역 damping 제거 (Fig. 6)                    #
    # ================================================================== #
    def rescale_neutron_moisture(self, theta, c_invariant, h=5.0):
        """
        footprint 내 불변 구역(건물·도로·물체 등)이 신호를 c_invariant 만큼
        고정 기여(damping)할 때, 관심 영역의 N(θ) 관계를 재조정한다 (Fig.6).

        관심 영역 신호 분율 = (1 - c_invariant) 이므로,
        유효 N0를 (1 - c_invariant)배로 낮춘 N–θ 곡선을 반환.

        Parameters
        ----------
        theta : float or array
        c_invariant : float
            불변 구역의 신호 기여 분율 (0~1). remote_field_contribution 등으로 산출.

        Returns
        -------
        N_rescaled : 재조정된 중성자 카운트
        """
        if self.config is not None:
            N0 = self.config.physics["neutron_moisture"]["N0"]
        else:
            N0 = 1950.0
        N0_eff = N0 * (1.0 - c_invariant)
        return self.physics.neutron_moisture_relationship(theta, h=h, N0=N0_eff)
