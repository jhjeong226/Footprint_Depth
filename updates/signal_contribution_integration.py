"""
signal_contribution.py 통합 가이드 (analyzer.py)
================================================

새 모듈 SignalContribution 은 검증된 physics(CRNPPhysics)를 받아 동작한다.
analyzer 에서 아래처럼 초기화하고, 일별 분석 결과에 선택적으로 덧붙인다.
"""

# ─────────────────────────────────────────────────────────────────────────
# 1) CRNPAnalyzer.__init__ 에 추가
# ─────────────────────────────────────────────────────────────────────────
#   from .signal_contribution import SignalContribution
#   ...
#   self.sigcon = SignalContribution(self.physics, self.config)


# ─────────────────────────────────────────────────────────────────────────
# 2) analyze_single_day(...) 끝부분, results 딕셔너리에 추가 (선택)
# ─────────────────────────────────────────────────────────────────────────
def append_signal_contribution(self, results, h=5.0):
    """일별 결과에 Schrön 2023 signal-contribution 지표 추가."""
    theta = results["predicted_swc"]
    press = results.get("pressure_hpa") or 1013.25
    Hveg = results.get("veg_height", 0.0)

    # (a) 실용적 footprint 거리 (Δθ=±10%, σN=1%) — wetting / drying
    R_wet = self.sigcon.practical_footprint_distance(
        theta, dtheta=+0.10, sigma_N=0.01, h=h,
        pressure_hpa=press, Hveg=Hveg)
    R_dry = self.sigcon.practical_footprint_distance(
        theta, dtheta=-0.10, sigma_N=0.01, h=h,
        pressure_hpa=press, Hveg=Hveg)

    # (b) 빠른 근사식 (Eq.15, R86 기반) — 비교/보고용
    R_approx = self.sigcon.practical_footprint_approx(
        theta, dtheta=0.05, h=h, pressure_hpa=press, Hveg=Hveg, use_R86=True)

    results.update({
        "practical_footprint_wet_m": R_wet,    # +10% 변화 감지 최대 거리
        "practical_footprint_dry_m": R_dry,    # -10% 변화 감지 최대 거리
        "practical_footprint_approx_m": R_approx,
    })
    return results


# ─────────────────────────────────────────────────────────────────────────
# 3) 단발성 분석 예시 — 특정 원거리 밭의 기여도
# ─────────────────────────────────────────────────────────────────────────
def example_remote_field(self):
    """
    예: 센서에서 80m 떨어진 관개 밭(θ=0.35)이, 주변(θ=0.15) 대비
        신호에 몇 % 기여하는가?
    """
    res = self.sigcon.remote_field_contribution(
        R=80, theta_main=0.15, theta_remote=0.35,
        h=5.0, pressure_hpa=1013.25, Hveg=0.3,
        geometry="cartesian")
    # res = {'c2': ..., 'c1': ..., 'theta_eff': ..., 'w': ...}
    return res


# ─────────────────────────────────────────────────────────────────────────
# 4) N(θ) 재조정 예시 — footprint 내 도로/건물 damping 제거
# ─────────────────────────────────────────────────────────────────────────
def example_rescale(self):
    """
    footprint 안에 신호의 12%를 고정 기여하는 도로가 있을 때,
    관심 토양 영역에 맞는 재조정 N0로 N(θ) 곡선을 다시 그린다.
    """
    c_road = 0.12   # remote_field_contribution 등으로 추정한 값
    N_rescaled = self.sigcon.rescale_neutron_moisture(
        theta=0.20, c_invariant=c_road, h=5.0)
    return N_rescaled
