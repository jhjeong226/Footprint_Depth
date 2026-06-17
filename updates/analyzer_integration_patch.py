"""
analyzer.py 통합 패치 (compute_contribution 부분)
==================================================

기존 흐름:
    R = sqrt(Xi^2 + Yi^2)
    R_eff = R * sP                      # 기압만 반영
    Wr = radial_intensity_function(R_eff, h, theta, bulk_density)

문제: R_eff가 기압(sP)만 반영. 완전한 footprint는 식생·습도·전제 토양수분까지
      r* 재스케일링에 들어가야 하고, Wr 내부는 r*를 받아야 함.

아래처럼 교체하세요.
"""

# ──────────────────────────────────────────────────────────────────────────
# BEFORE (기존)
# ──────────────────────────────────────────────────────────────────────────
#   R = np.sqrt(Xi**2 + Yi**2)
#   mask_r = R <= max_extent
#   sP = (pressure_hpa / P0) ** alpha_p   (or 1.0)
#   R_eff = R * sP
#   Wr = self.physics.radial_intensity_function(R_eff, h=h, theta=theta,
#                                                bulk_density=bulk_density)

# ──────────────────────────────────────────────────────────────────────────
# AFTER (완전판)
# ──────────────────────────────────────────────────────────────────────────
def compute_contribution_snippet(self, Xi, Yi, theta_map, Hveg, h,
                                  pressure_hpa, P0, max_extent,
                                  bulk_density):
    import numpy as np

    R = np.sqrt(Xi**2 + Yi**2)
    mask_r = R <= max_extent

    # 대표 토양수분 (footprint 파라미터 함수는 스칼라 조건을 가정)
    theta_repr = float(np.nanmean(theta_map[mask_r]))

    # --- r* 재스케일링: 기압 + 식생 + 전제 토양수분 (Schrön 2017) ---
    R_star = self.physics.rescale_distance(
        R, pressure_hpa=pressure_hpa, Hveg=float(Hveg),
        theta=theta_repr, P0=P0
    )

    # 기압 스케일 팩터(보고용) — rescale_distance 안에 이미 반영됨
    if pressure_hpa is None or not np.isfinite(pressure_hpa):
        sP = 1.0
    else:
        sP = 0.4922 / (0.86 - np.exp(-pressure_hpa / P0))

    # --- 완전한 Wr (r* 입력, 대표 θ·h로 파라미터 함수 평가) ---
    Wr = self.physics.radial_intensity_function(
        R_star, h=h, theta=theta_repr, bulk_density=bulk_density
    )

    # --- 순수 커널 K (Wr/r, 정규화) : 공간 가중만 ---
    # 주의: Schrön Eq.9는 그리드에서 Wr/r 로 면적효과 보정.
    #       r=0 특이점 방지 위해 (R + dr/2) 사용 권장.
    K = np.zeros_like(R, dtype=float)
    mK = mask_r & np.isfinite(Wr) & (R > 0)
    K[mK] = Wr[mK] / R[mK]
    Ks = np.nansum(K)
    if Ks > 0:
        K /= Ks

    # --- θ-의존 변조(공간적): 식생 보정 × 국소 중성자 생산 N(θ) ---
    fveg = self.physics.vegetation_correction_factor(theta_map, Hveg)
    N0 = self.config.physics['neutron_moisture']['N0']
    N_theta = self.physics.neutron_moisture_relationship(theta_map, h=h, N0=N0)

    # --- 최종 기여도 C = Wr/r × fveg × N(θ) ---
    C = np.zeros_like(R, dtype=float)
    mask = mask_r & np.isfinite(theta_map) & np.isfinite(N_theta) & (R > 0)
    C[mask] = Wr[mask] * fveg[mask] * N_theta[mask] / R[mask]
    s = np.nansum(C)
    if s > 0:
        C /= s

    return C, float(sP), K


# ──────────────────────────────────────────────────────────────────────────
# R86 보고: 그리드 기반(max_extent 의존) + 해석적(독립) 둘 다 산출
# ──────────────────────────────────────────────────────────────────────────
#   R86_analytical = self.physics.calculate_analytical_R86(
#       theta=predicted_swc, bulk_density=bulk_density, h=h,
#       pressure_hpa=pressure_hpa, Hveg=veg_height)
#
# 이제 R86_analytical 이 토양수분에 따라 실제로 변함:
#   dry(θ=0.10) → ~204 m,  wet(θ=0.40) → ~121 m
# → 기존 "R86가 토양수분에 무관하게 동일" 버그 해결.
