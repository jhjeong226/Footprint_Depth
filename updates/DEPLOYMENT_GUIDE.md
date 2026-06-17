# 전체 적용 가이드 — 파일 배치 및 실제 사용 방법

지금까지 만든 모든 코드를 실제 프로젝트(`Footprint_Depth`)에서 동작시키기 위한
배치 위치와 통합 단계입니다. 순서대로 진행하면 됩니다.

---

## 0. 파일 → 위치 매핑 요약

| 파일 | 최종 위치 | 작업 |
|------|-----------|------|
| `physics.py` | `src/core/physics.py` | **전체 교체** |
| `physics_config_block.yaml` | `config/default.yaml`, `config/sites/HC.yaml`, `config/sites/PC.yaml` | `physics:` 블록 **병합** |
| `signal_contribution.py` | `src/core/signal_contribution.py` | **신규 추가** |
| `panel_06_practical_footprint.py` | `src/plotting/panel_06_practical_footprint.py` | **신규 추가** |
| (analyzer 수정) | `src/core/analyzer.py` | **부분 수정** (아래 3, 5단계) |
| (init 수정) | `src/core/__init__.py`, `src/plotting/__init__.py` | **import 추가** |

> 참고: `analyzer_integration_patch.py`, `signal_contribution_integration.py`는
> 통합 방법을 보여주는 **참조용**입니다. 그 안의 코드 조각을 실제 `analyzer.py`에 반영하세요.

---

## 1. physics.py 교체

`src/core/physics.py` 를 새 `physics.py` 로 통째로 교체합니다.
(검증 완료: Schrön 2017 Table A1과 43개 파라미터 전부 일치, neptoon과 점값 일치)

**주의 — 함수 시그니처 변경점:**
- `radial_intensity_function(r, h, theta, bulk_density)` : 이제 `r`은 **재스케일된 r***를 받습니다.
- `calculate_penetration_depth(theta, bulk_density, r)` : `r`은 r* 사용 권장.
- 신규: `rescale_distance(...)`, `calculate_analytical_R86(...)`, `radial_intensity_function_approx(...)`

---

## 2. config YAML 병합

세 파일(`default.yaml`, `sites/HC.yaml`, `sites/PC.yaml`)의 `physics:` 블록을
`physics_config_block.yaml` 내용으로 교체합니다. 단:
- `neutron_moisture.N0` 는 사이트별 값 유지 (HC: 1753, PC: 2400, default: 1950)
- `radial_weighting_full` (43개 계수) **신규 추가**
- `depth_weighting` 키 변경: 기존 `p3, p4` 삭제 → `c_num=20.0, c_den=0.0429, Lr=100.0` 추가, `p2: 0.86655 → 0.96655`

---

## 3. analyzer.py — footprint 계산부 수정

`compute_contribution`(또는 동등 함수)에서 `R_eff = R * sP` 를 r* 재스케일링으로 교체:

```python
# BEFORE: R_eff = R * sP ; Wr = self.physics.radial_intensity_function(R_eff, ...)

# AFTER:
theta_repr = float(np.nanmean(theta_map[mask_r]))
R_star = self.physics.rescale_distance(
    R, pressure_hpa=pressure_hpa, Hveg=float(Hveg),
    theta=theta_repr, P0=P0)
Wr = self.physics.radial_intensity_function(R_star, h=h, theta=theta_repr)

# 기여도: Wr/r × fveg × N(θ)  (r=0 특이점은 R>0 마스크로 처리)
fveg = self.physics.vegetation_correction_factor(theta_map, Hveg)
N_theta = self.physics.neutron_moisture_relationship(theta_map, h=h, N0=N0)
C = np.zeros_like(R); m = mask_r & (R>0) & np.isfinite(theta_map) & np.isfinite(N_theta)
C[m] = Wr[m] * fveg[m] * N_theta[m] / R[m]
C /= np.nansum(C)
```

`R86_analytical` 도 이제 토양수분에 따라 변합니다:
```python
R86_analytical = self.physics.calculate_analytical_R86(
    theta=predicted_swc, bulk_density=bulk_density, h=h,
    pressure_hpa=pressure_hpa, Hveg=veg_height)
```
(자세한 내용은 `analyzer_integration_patch.py` 참조)

---

## 4. signal_contribution.py 추가 + import

1. `signal_contribution.py` 를 `src/core/` 에 복사.
2. `src/core/__init__.py` 에 추가:
```python
from .signal_contribution import SignalContribution
__all__ += ['SignalContribution']
```

---

## 5. analyzer.py — SignalContribution 연결

`CRNPAnalyzer.__init__` 에 추가:
```python
from .signal_contribution import SignalContribution
self.sigcon = SignalContribution(self.physics, self.config)
```

`analyze_single_day` 결과에 실용적 footprint 추가 (선택):
```python
theta = results["predicted_swc"]; press = results.get("pressure_hpa") or 1013.25
Hveg = results.get("veg_height", 0.0)
results["practical_footprint_wet_m"] = self.sigcon.practical_footprint_distance(
    theta, dtheta=+0.10, sigma_N=0.01, h=5.0, pressure_hpa=press, Hveg=Hveg)
results["practical_footprint_dry_m"] = self.sigcon.practical_footprint_distance(
    theta, dtheta=-0.10, sigma_N=0.01, h=5.0, pressure_hpa=press, Hveg=Hveg)
```

---

## 6. 시각화 패널 추가 + import

1. `panel_06_practical_footprint.py` 를 `src/plotting/` 에 복사.
2. `src/plotting/__init__.py` 에 추가:
```python
from .panel_06_practical_footprint import plot_panel_practical_footprint
__all__ += ['plot_panel_practical_footprint']
```
3. `save_all_panels(config, analyzer, results, out_dir, prefix)` 끝부분에 추가:
```python
# Panel 6: Practical footprint (Schrön 2023)
plot_panel_practical_footprint(
    results,
    os.path.join(out_dir, f"{prefix}_{d}_06_practical_footprint.png"),
    config,
    analyzer.sigcon,        # ← SignalContribution 인스턴스 전달
    h=results.get("abs_humidity") or 5.0)
```

> 다른 패널과 달리 이 패널은 `sigcon` 인자가 추가로 필요합니다.
> `save_all_panels`가 `analyzer`를 이미 받으므로 `analyzer.sigcon`을 넘기면 됩니다.

---

## 7. 동작 확인

```bash
python main.py   # 또는 기존 실행 스크립트
```
출력 폴더에 `..._06_practical_footprint.png` 가 생성되고,
콘솔/결과에 `practical_footprint_wet_m`, `..._dry_m` 이 찍히면 성공입니다.

기대 거동 (θ=20%, h=5, P=1013 기준 예시):
- R86 ≈ 170 m,  wet(+10%) ≈ 85 m,  dry(−10%) ≈ 144 m
- dry > wet (비대칭 ~1.5–2.0), 건조할수록 모든 반경 확대

---

## 8. 검증 근거 (이미 완료)

| 항목 | 검증 결과 |
|------|-----------|
| Wr 파라미터 | Schrön 2017 Table A1과 43개 전부 일치 |
| Wr 점값 | neptoon v0.14와 rtol=1e-6 일치 |
| R86 (θ별) | dry 210m → wet 121m, 논문 문헌값 일치 |
| D86 | 논문 곡선 (dry~39, wet~17cm) 일치 |
| 원거리 기여도 c2 | R=57m→13.8% (논문 14.7%), R=207m→2.8% (논문 2.9%) |
| 실용적 footprint R | 논문 Table 1과 평균 ~3% 편차 |
| dry/wet 비대칭 | 1.68배 (논문 1.5–2.0) |

---

## 의존성
`scipy.optimize.brentq` 사용 (실용적 footprint 수치해 / N 역변환).
이미 프로젝트가 scipy를 쓰므로 추가 설치 불필요.
```python
# requirements: numpy, scipy, matplotlib, pyyaml  (기존과 동일)
```
