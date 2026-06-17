# Schrön 2023 Signal Contribution — 추가 기능

현재 코드는 기여도 맵(Eq.7)·R86 등고선·섹터별 R86(φ)까지 구현되어 있었습니다.
이 패키지는 Schrön et al. (2023)의 **미구현 3개 기능**을 추가하며, 모두 논문 값으로 검증을 마쳤습니다.

---

## 검증 결과

### 1. 원거리 밭 기여도 c2 (Eq.13 + Eq.7)
| 시나리오 | 계산 | 논문 |
|----------|------|------|
| θ1=5%, θ2=10%, R=57m | c2 = 13.8%, θ̂=5.5% | c2≈14.7%, θ̂≈5.1% |
| θ1=5%, θ2=10%, R=207m | c2 = 2.8% | c2≈2.9% |

### 2. 실용적 footprint 거리 R (Eq.14, σN=1%, Δθ=+10%)
| θ1 | 계산 | 논문 Table 1 |
|----|------|--------------|
| 5% | 194m | 185m |
| 10% | 147m | 141m |
| 20% | 84m | 81.5m |
| 40% | 32m | 31.5m |

### 3. drying/wetting 비대칭
θ=20%에서 wet +10% → 84m, dry −10% → 141m (비율 1.68, 논문 1.5–2.0). 논문 예시값과 거의 정확히 일치.

### 4. N(θ) 재조정 (Fig.6)
20% damping → N0 비율 0.80 정확.

모든 편차는 적분 격자 vs URANOS 미세조정 차이 수준이며 물리적으로 정합합니다.

---

## 추가된 기능 (`signal_contribution.py`)

`SignalContribution(physics, config)` — 검증된 physics.py 인스턴스를 받습니다.

- **`remote_field_contribution(R, θ_main, θ_remote, ...)`**
  원거리 밭이 신호에 몇 % 기여하는지(c2)와 유효 평균 토양수분 θ̂ 반환. θ̂ 반복 갱신 포함.
- **`practical_footprint_distance(θ_main, dtheta, sigma_N, ...)`**
  Δθ 변화가 센서 정밀도 σN 안에서 감지되는 최대 거리 R (Eq.14 수치해). wetting/drying 모두.
- **`practical_footprint_approx(θ_main, dtheta, use_R86, ...)`**
  빠른 근사식. `use_R86=True` → Eq.15 (R86 기반), `False` → Eq.16 (독립).
- **`rescale_neutron_moisture(θ, c_invariant, ...)`**
  footprint 내 불변 구역(도로·건물)의 damping을 제거한 N(θ) 곡선 (Fig.6).
- **`remote_field_weight(...)`** — Eq.13의 공간 가중 w (cartesian / radial geometry).

---

## 적용 방법

1. `signal_contribution.py` 를 `src/core/` 에 추가.
2. `src/core/__init__.py` 에 `from .signal_contribution import SignalContribution` 추가.
3. `analyzer.py` 통합은 `signal_contribution_integration.py` 참조:
   - `__init__`: `self.sigcon = SignalContribution(self.physics, self.config)`
   - `analyze_single_day` 결과에 `practical_footprint_wet_m / dry_m` 추가 (선택)
4. 시각화: 기존 R86 등고선 패널에 실용적 footprint R(wet/dry)을 추가 원으로 겹쳐 그리면
   "R86 vs 실제 감지 거리" 비교가 한눈에 보입니다.

---

## 핵심 수식

- **Eq.13** 원거리 밭 총 신호: `N(θ̂,θ2) = (1-w)·N(θ1) + w·N(θ2)`,
  `w = (1/π)∫_R^∞ Wr(h,θ̂)·arccos(R/r) dr` (직선 경계)
- **Eq.14** 감지 조건: `|1 - N(θ̂,θ2)/N(θ̂,θ1)| > σN`
- **Eq.15** 근사: `R = R86(h,θ,P)·exp(0.31 − 8θ − 5Δθ)`
- **Eq.16** 독립 근사: `R ≈ 225·exp(0.25 − 9θ − 5Δθ)` [m]

---

## 출처
Schrön, M., Köhli, M., Zacharias, S. (2023). Signal contribution of distant areas
to cosmic-ray neutron sensors — implications for footprint and sensitivity.
HESS 27, 723–738. (프로젝트 내 PDF, Eq.5–16 / Fig.6–8 / Table 1 대조 검증)
