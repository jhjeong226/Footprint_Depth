"""
CRNP 물리 방정식 (완전판)

수평 가중 함수 Wr(r, h, θ, P, Hveg):
    Köhli et al. (2015), Schrön et al. (2017)의 완전한 F1–F4 (8-파라미터) 공식.
    토양수분/습도/기압/식생에 따라 footprint가 동적으로 변함 (R86 ≈ 130–240 m).
    구간별 형태: r≤1m, 1<r<50m, r≥50m (Schrön 2017 supplement과 동일).

침투 깊이 D86, 깊이 가중 Wd:
    Köhli et al. (2015), Schrön et al. (2017) / Wang et al. (2023).

검증: neptoon(v0.14) 및 Schrön 2017 lookup table과 R86 편차 < 5%,
      θ=0.05→~210m, θ=0.40→~121m (dry>wet) 거동 재현.
"""
import numpy as np


class CRNPPhysics:
    """CRNP 물리 계산 클래스"""

    def __init__(self, config):
        self.config = config
        self.physics = config.physics

    # ------------------------------------------------------------------ #
    # 거리 재스케일링 r* (기압·식생·전제 토양수분)                          #
    #   Köhli(2015)/Schrön(2017): Wr은 r*에 대해 평가됨                    #
    # ------------------------------------------------------------------ #
    def rescale_distance(self, r, pressure_hpa=1013.25, Hveg=0.0,
                         theta=0.10, P0=1013.25):
        """
        물리 거리 r(m) -> 재스케일 거리 r*(m).

        F_p   : 기압 영향 (Schrön 2017)
        F_veg : 식생 영향
        """
        if pressure_hpa is None or not np.isfinite(pressure_hpa):
            pressure_hpa = P0
        F_p = 0.4922 / (0.86 - np.exp(-pressure_hpa / P0))
        F_veg = 1.0 - 0.17 * (1.0 - np.exp(-0.41 * Hveg)) * (
            1.0 + np.exp(-9.25 * theta)
        )
        return r * F_p * F_veg

    # ------------------------------------------------------------------ #
    # 수평 가중 함수 Wr — 완전한 Köhli(2015)/Schrön(2017) F1–F4            #
    # ------------------------------------------------------------------ #
    def radial_intensity_function(self, r, h=5.0, theta=0.10,
                                  bulk_density=1.4):
        """
        Wr(r*, h, θ) — 토양수분/습도 의존적 방사형 가중 함수.

        Based on: Köhli et al. (2015) Appendix A, Schrön et al. (2017) Appendix A.

        Parameters
        ----------
        r : float or ndarray
            재스케일된 거리 r* (m). rescale_distance()로 변환 후 전달.
        h : float
            절대 습도 (g/m³). 논문 표기 x.
        theta : float
            대표 체적 토양수분 (m³/m³), 0.02–0.50 권장. 논문 표기 y.
            (배열이 들어오면 평균값을 대표값으로 사용 — 파라미터 함수는
             스칼라 조건을 가정하므로 공간 변조는 분리 처리.)

        Returns
        -------
        Wr : float or ndarray
            방사형 가중값 (정규화 전).
        """
        # 파라미터 함수는 스칼라 (h, θ) 조건을 받음
        if hasattr(theta, "__len__"):
            y = float(np.nanmean(theta))
        else:
            y = float(theta)
        x = float(h)

        # 입력 범위 보호 (논문 유효 구간)
        y = float(np.clip(y, 0.01, 0.49))
        if x > 29:
            x = 29.0

        p = self.physics["radial_weighting_full"]

        # --- 근거리(near) 파라미터 함수 A0..A3 -------------------------- #
        A0 = (p["a00"] * (1 + p["a03"] * x) * np.exp(-p["a01"] * y)
              + p["a02"] * (1 + p["a05"] * x) - p["a04"] * y)
        A1 = (((-p["a10"] + p["a14"] * x)
               * np.exp(-p["a11"] * y / (1 + p["a15"] * y)) + p["a12"])
              * (1 + x * p["a13"]))
        A2 = (p["a20"] * (1 + p["a23"] * x) * np.exp(-p["a21"] * y)
              + p["a22"] - p["a24"] * y)
        A3 = (p["a30"] * np.exp(-p["a31"] * y) + p["a32"]
              - p["a33"] * y + p["a34"] * x)

        # --- 원거리(far) 파라미터 함수 B0..B3 --------------------------- #
        B0 = ((p["b00"] - p["b01"] / (p["b02"] * y + x - 0.13))
              * (p["b03"] - y) * np.exp(-p["b04"] * y)
              - p["b05"] * x * y + p["b06"])
        B1 = p["b10"] * (x + p["b11"]) + p["b12"] * y
        B2 = ((p["b20"] * (1 - p["b26"] * x)
               * np.exp(-p["b21"] * y * (1 - x * p["b24"]))
               + p["b22"] - p["b25"] * y) * (2 + x * p["b23"]))
        B3 = (((-p["b30"] + p["b34"] * x)
               * np.exp(-p["b31"] * y / (1 + p["b35"] * x + p["b36"] * y))
               + p["b32"]) * (2 + x * p["b33"]))

        r = np.asarray(r, dtype=float)
        scalar = (r.ndim == 0)
        r = np.atleast_1d(r)
        W = np.zeros_like(r, dtype=float)

        near0 = r <= 1.0                       # r ≤ 1 m
        near = (r > 1.0) & (r < 50.0)          # 1 < r < 50 m
        far = r >= 50.0                        # r ≥ 50 m

        W[near0] = (A0 * np.exp(-A1 * r[near0]) + A2 * np.exp(-A3 * r[near0])) \
            * (1.0 - np.exp(-3.7 * r[near0]))
        W[near] = A0 * np.exp(-A1 * r[near]) + A2 * np.exp(-A3 * r[near])
        W[far] = B0 * np.exp(-B1 * r[far]) + B2 * np.exp(-B3 * r[far])

        # 음수/비유한 방지
        W = np.where(np.isfinite(W), W, 0.0)
        W = np.clip(W, 0.0, None)

        return float(W[0]) if scalar else W

    def radial_intensity_function_approx(self, r):
        """
        단순 근사식 W*r (Schrön 2023 Eq.2) — θ/h 비의존, 빠른 비교용.
        """
        r = np.asarray(r, dtype=float)
        return (30.0 * np.exp(-r / 1.6) + np.exp(-r / 100.0)) \
            * (1.0 - np.exp(-3.7 * r))

    # ------------------------------------------------------------------ #
    # 침투 깊이 D86 — Köhli(2015)/Schrön(2017) 검증판                      #
    # ------------------------------------------------------------------ #
    def calculate_penetration_depth(self, theta, bulk_density=1.4, r=0):
        """
        침투 깊이 D86 (cm). 86% 중성자가 도달한 깊이.

        D86 = (1/ρb) · [p0 + p1·(p2 + exp(-r/Lr))·(c_num + θ)/(c_den + θ)]

        검증된 계수 (neptoon v0.14 / Schrön 2017):
            p0=8.321, p1=0.14249, p2=0.96655,
            c_num=20.0, c_den=0.0429, Lr=100 (exp(-0.01·r))
        → θ=0.10,r=0,ρb=1.22: D86≈39cm; θ=0.40: ≈17cm (논문 곡선 일치)
        """
        params = self.physics["depth_weighting"]
        p0 = params["p0"]
        p1 = params["p1"]
        p2 = params["p2"]
        c_num = params["c_num"]
        c_den = params["c_den"]
        Lr = params.get("Lr", 100.0)

        theta = np.asarray(theta, dtype=float)
        denom = c_den + theta
        denom = np.where(denom == 0, 1e-10, denom)

        D = (1.0 / bulk_density) * (
            p0 + p1 * (p2 + np.exp(-r / Lr)) * (c_num + theta) / denom
        )
        return float(D) if np.isscalar(theta) else D

    def depth_weighting_function(self, d, theta, bulk_density=1.4, r=0):
        """
        깊이 가중 함수 Wd(d) = exp(-2·d / D86).  Wang(2023) Eq.11.
        """
        D = self.calculate_penetration_depth(theta, bulk_density, r)
        return np.exp(-2.0 * d / D)

    # ------------------------------------------------------------------ #
    # 식생 보정 / 중성자-토양수분 관계 (기존 유지)                          #
    # ------------------------------------------------------------------ #
    def vegetation_correction_factor(self, theta, Hveg):
        v = self.physics["vegetation"]
        c1, c2, c3 = v["c1"], v["c2"], v["c3"]
        return 1.0 - c1 * (1.0 - np.exp(-c2 * Hveg)) * (1.0 + np.exp(-c3 * theta))

    def neutron_moisture_relationship(self, theta, h=5.0, N0=1950):
        params = self.physics["neutron_moisture"]
        p1, p2, p3 = params["p1"], params["p2"], params["p3"]
        p4, p5, p6 = params["p4"], params["p5"], params["p6"]
        p7, p8 = params["p7"], params["p8"]
        N = N0 * (
            (p1 + p2 * theta) / (p1 + theta) * (p3 + p4 * h + p5 * h**2)
            + np.exp(-p6 * theta) * (p7 + p8 * h)
        )
        return N

    # ------------------------------------------------------------------ #
    # 해석적 R86 — max_extent 독립, 완전한 Wr 적분                          #
    # ------------------------------------------------------------------ #
    def calculate_analytical_R86(self, theta, bulk_density=1.4, h=5.0,
                                 pressure_hpa=1013.25, Hveg=0.0,
                                 target=0.86, r_max=700.0, dr=0.1, P0=1013.25):
        """
        Wr(r*)을 물리 거리 r에 대해 적분하여 R86 산출 (Eq.4).
        max_extent 그리드와 무관한 진짜 footprint 반경.
        """
        if hasattr(theta, "__len__"):
            theta = float(np.nanmean(theta))
        r = np.arange(dr, r_max, dr)
        rstar = self.rescale_distance(r, pressure_hpa, Hveg, theta, P0)
        w = self.radial_intensity_function(rstar, h=h, theta=theta,
                                           bulk_density=bulk_density)
        cum = np.cumsum(w)
        if cum[-1] <= 0:
            return np.nan
        cum /= cum[-1]
        idx = np.searchsorted(cum, target)
        return float(r[min(idx, len(r) - 1)])
