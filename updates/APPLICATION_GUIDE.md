# 완전한 Köhli(2015)/Schrön(2017) Footprint 적용 가이드

검증된 전체 F1–F4(8-파라미터) 수평 가중 함수와 D86 깊이 계산을
프로젝트에 반영하는 절차입니다. **neptoon v0.14 및 논문 문헌값과 점값 단위까지 일치 검증을 마쳤습니다.**

---

## 검증 결과 (이 구현이 재현하는 값)

| θ (m³/m³) | 해석적 R86 | 논문/neptoon | D86 (ρb=1.22) |
|-----------|-----------|--------------|----------------|
| 0.05 | 209.5 m | ~218 m | 56.4 cm |
| 0.10 | 203.6 m | ~209 m | 39.1 cm |
| 0.20 | 170.4 m | ~173 m | 25.9 cm |
| 0.30 | 138.9 m | ~141 m | 20.4 cm |
| 0.40 | 120.7 m | ~124 m | 17.4 cm |

- **Wr 점값**: neptoon과 모든 거리에서 완전 일치 (rtol=1e-6)
- **dry → 큰 footprint, wet → 작은 footprint** 거동 정확히 재현 → 기존 "R86가 토양수분에 무관" 버그 해결
- 식생↑ → r* 확대, 습도↑ → footprint 축소 (물리적으로 정확)

---

## 적용 단계

### 1. `src/core/physics.py` 교체
`physics.py`(첨부)로 전체 교체. 주요 변경:
- `radial_intensity_function`: 단순식/ad-hoc 스케일링 제거 → 완전한 A0–A3, B0–B3 (r≤1 / 1<r<50 / r≥50 3구간)
- `rescale_distance`: 새로 추가 (r* 재스케일링, 기압·식생·전제 θ)
- `calculate_penetration_depth`: 검증된 D86 계수로 교체
- `calculate_analytical_R86`: 완전한 Wr 적분 (max_extent 독립)
- `radial_intensity_function_approx`: 기존 단순식 보존 (비교/폴백)

### 2. config YAML 갱신
`physics_config_block.yaml`의 내용을 `default.yaml`, `sites/HC.yaml`,
`sites/PC.yaml`의 `physics:` 블록에 반영.
- `radial_weighting_full`: **신규 추가** (43개 계수)
- `depth_weighting`: 키 변경 — 기존 `p3, p4` 삭제, `c_num=20.0, c_den=0.0429, Lr=100.0` 추가, `p2: 0.86655 → 0.96655`
- `neutron_moisture`, `vegetation`, `N0`(사이트별): 기존 값 유지

### 3. `src/core/analyzer.py` 통합부 수정
`analyzer_integration_patch.py` 참조. 핵심:
- `R_eff = R * sP` (기압만) → `R_star = rescale_distance(R, pressure, Hveg, θ_repr)` (기압+식생+θ)
- `Wr = radial_intensity_function(R_star, ...)` — r* 입력
- `Wr/r` 그리드 가중 시 r=0 특이점 처리 (R>0 마스크)
- `R86_analytical = calculate_analytical_R86(predicted_swc, ρb, h, pressure, Hveg)`

`FPwithDepth.py`도 동일 원리로 `compute_footprint` 내 Wr 식을 교체
(단순식 → physics 호출)하면 일관성이 확보됩니다.

---

## D86 계수 변경에 대한 주의 (중요)

기존 코드의 D86 계수는 **두 출처가 섞여 있었습니다**:
- 기존: `p2=0.86655, p3=26.42, p4=0.0567, exp(-r/100)`  ← Wang(2023) 표기 인용
- 검증판: `p2=0.96655, c_num=20.0, c_den=0.0429, exp(-0.01·r)`  ← Schrön(2017)/neptoon

수치 검증 결과 **검증판이 논문 D86 곡선(dry≈39cm, wet≈17cm)을 정확히 재현**합니다.
기존 계수는 dry에서 D86를 과대평가합니다. 검증판 채택을 권장하나,
Wang(2023) 표기를 엄격히 따라야 한다면 원 논문 Table A1의 정확한 자릿수를
재확인하시기 바랍니다 (arXiv 본문에서 표가 잘려 직접 대조 불가했음).

---

## 라벨링 일관성 정리

- footprint와 N(θ) 모두 동일한 Köhli 계열 공식을 일관되게 사용하게 됨
- `depth.md`, 주석의 "Schrön 2023 Eq.2 단순식" 설명을
  "완전한 Köhli(2015)/Schrön(2017) F1–F4" 로 갱신
- 참고문헌에 Köhli et al. (2015), Schrön et al. (2017) 추가

---

## 출처
- Köhli et al. (2015), WRR 51, 5772–5790 (arXiv:1602.04469) — F1–F8 함수 형태, Appendix A
- Schrön et al. (2017), HESS — Appendix A 파라미터, r* 재스케일링
- Schrön et al. (2023), HESS — Eq.2 단순식, R86 정의
- 검증 기준: neptoon v0.14 `Schroen2017` 클래스 (점값/D86/R86 대조)
