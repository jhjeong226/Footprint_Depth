# CRNP 유효깊이 (Effective Depth) 산정 이론 및 프로세스

## 1. 개요

우주선 중성자 탐지법(CRNP)에서 **유효깊이(Effective Depth)**는 중성자 신호가 주로 유래하는 토양 깊이를 의미한다. 이 코드는 **Wang et al. (2023)**의 이론을 기반으로 유효깊이를 계산한다.

중성자는 토양 표면 가까이에서 대부분 생성되며, 깊이가 증가할수록 기여도가 지수적으로 감소한다.

---

## 2. 깊이 가중 함수 (Depth Weighting Function)

### 2.1 기본 수식

깊이 가중 함수 $W_d(d)$는 깊이 $d$에 위치한 토양이 중성자 신호에 기여하는 정도를 나타낸다.

**수식 (Wang et al. 2023, Equation 6):**

$$W_d(d) = e^{-2d/D}$$

여기서:
- $d$: 토양 깊이 (cm)
- $D$: 침투 깊이 (penetration depth, cm)

### 2.2 물리적 의미

- 지수 함수적 감쇠: 깊이가 증가할수록 기여도가 급격히 감소
- 계수 2: 중성자가 토양 내로 들어갔다가 다시 나오는 왕복 경로 반영
- $D$: 기여도가 $1/e$ (약 37%)로 감소하는 특성 깊이

---

## 3. 침투 깊이 (Penetration Depth) 계산

### 3.1 계산 수식

침투 깊이 $D$는 토양수분, 토양 밀도, CRNP로부터의 거리에 따라 결정된다.

**수식 (Wang et al. 2023, Equation 6):**

$$D = \frac{1}{\rho_b} \left( p_0 + p_1 \cdot \left( p_2 + e^{-r/100} \cdot \frac{p_3 + \theta}{p_4 + \theta} \right) \right)$$

여기서:
- $\rho_b$: 토양 건조 밀도 (bulk density, g/cm³)
- $\theta$: 체적 토양수분 (m³/m³)
- $r$: CRNP로부터의 수평 거리 (m)
- $p_0, p_1, p_2, p_3, p_4$: 피팅 파라미터

### 3.2 파라미터 값

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| $p_0$ | 8.321 | 기본 침투 깊이 스케일 |
| $p_1$ | 0.14249 | 거리 의존성 계수 |
| $p_2$ | 0.86655 | 거리 무관 항 |
| $p_3$ | 26.42 | 토양수분 보정 항 (분자) |
| $p_4$ | 0.0567 | 토양수분 보정 항 (분모) |

### 3.3 침투 깊이의 변화

#### 토양수분 의존성

| 조건 | 침투 깊이 | 설명 |
|------|-----------|------|
| 건조 토양 ($\theta$ 낮음) | 증가 (깊음) | 수소 원자가 적어 중성자 감쇠 작음 |
| 습윤 토양 ($\theta$ 높음) | 감소 (얕음) | 수소 원자가 많아 중성자 감쇠 큼 |

#### 거리 의존성

$$e^{-r/100}$$

- $r = 0$ (센서 바로 아래): 거리 항 = 1 (최대 영향)
- $r = 100$ m: 거리 항 ≈ 0.37
- $r = 200$ m: 거리 항 ≈ 0.14
- 원거리에서는 거리 의존성 감소

#### 토양 밀도 의존성

$$D \propto \frac{1}{\rho_b}$$

- 밀도가 높은 토양: 침투 깊이 감소
- 밀도가 낮은 토양: 침투 깊이 증가

---

## 4. D86 계산 (86% 누적 기여 깊이)

### 4.1 정의

$D_{86}$은 전체 중성자 신호의 86%가 유래하는 토양 깊이이다.

**수식:**

$$\int_0^{D_{86}} W_d(z) \, dz = 0.86 \times \int_0^{\infty} W_d(z) \, dz$$

### 4.2 계산 방법

```python
def compute_depth_profile(theta, bulk_density, r=0, max_depth=100):
    depths = np.linspace(0, max_depth, n_points)

    # 침투 깊이 계산
    D = calculate_penetration_depth(theta, bulk_density, r)

    # 가중치 계산
    weights = np.exp(-2 * depths / D)
    weights_norm = weights / np.trapz(weights, depths)

    # 누적 분포
    cumulative = [np.trapz(weights_norm[:i+1], depths[:i+1])
                  for i in range(len(depths))]

    # D86 찾기
    D86 = depths[np.searchsorted(cumulative, 0.86)]
    return D86
```

### 4.3 예시 계산 결과

**조건**: $\rho_b = 1.22$ g/cm³, $r = 0$ m

| 토양수분 ($\theta$) | 침투 깊이 $D$ (cm) | $D_{86}$ (cm) |
|---------------------|-------------------|---------------|
| 0.10 | ~20-25 | ~25-30 |
| 0.20 | ~15-20 | ~20-25 |
| 0.30 | ~10-15 | ~15-20 |
| 0.40 | ~8-12 | ~12-15 |

---

## 5. 깊이별 기여도 분석

### 5.1 레이어별 기여도 계산

코드는 깊이를 4개 레이어로 나누어 각 레이어의 신호 기여도를 계산한다.

```python
layers = [(0, 10), (10, 20), (20, 40), (40, 100)]  # cm

for d_start, d_end in layers:
    idx_layer = (depths >= d_start) & (depths <= d_end)
    contrib = np.trapz(weights_norm[idx_layer], depths[idx_layer])
    layer_contributions.append(contrib)
```

### 5.2 예시 결과

**조건**: $\theta = 0.20$ m³/m³, $\rho_b = 1.22$ g/cm³

| 레이어 | 깊이 범위 | 예상 기여도 |
|--------|-----------|-------------|
| Layer 1 | 0-10 cm | ~45-55% |
| Layer 2 | 10-20 cm | ~25-35% |
| Layer 3 | 20-40 cm | ~10-15% |
| Layer 4 | 40-100 cm | ~2-5% |

---

## 6. 수평-수직 통합 가중 함수

### 6.1 3차원 가중 함수

전체 기여도는 수평 가중 함수와 깊이 가중 함수의 곱으로 표현된다.

$$W(r, d) = W_r(r) \times W_d(d, \theta(r))$$

여기서:
- $W_r(r)$: 수평 가중 함수 (footprint.md 참조)
- $W_d(d, \theta)$: 깊이 가중 함수
- $\theta(r)$: 거리에 따른 토양수분 분포

### 6.2 가중 평균 토양수분

CRNP가 측정하는 가중 평균 토양수분:

$$\bar{\theta} = \frac{\int_0^{R_{max}} \int_0^{d_{max}} W(r,d) \cdot \theta(r,d) \, r \, dr \, dd}{\int_0^{R_{max}} \int_0^{d_{max}} W(r,d) \cdot r \, dr \, dd}$$

---

## 7. 감쇠 길이 (Attenuation Length)

### 7.1 정의

감쇠 길이 $L$은 중성자 신호가 $1/e$로 감소하는 거리를 나타낸다.

### 7.2 계산 수식

$$L = \frac{1}{\alpha + \beta \cdot \theta_g}$$

여기서:
- $\alpha = 0.006$
- $\beta = 0.012$
- $\theta_g = \theta \cdot \rho_b$ (중량 토양수분)

### 7.3 제한 범위

$$L = \text{clip}(L, 20, 200) \text{ m}$$

---

## 8. 거리에 따른 유효깊이 변화

### 8.1 개념

CRNP에서 멀어질수록 유효깊이가 변한다.

```python
def calculate_penetration_depth(theta, bulk_density, r):
    term = (p3 + theta) / (p4 + theta)
    D = (1.0 / bulk_density) * (p0 + p1 * (p2 + np.exp(-r / 100.0) * term))
    return D
```

### 8.2 거리별 침투 깊이 변화

$$D(r) = \frac{1}{\rho_b} \left( p_0 + p_1 \cdot p_2 + p_1 \cdot e^{-r/100} \cdot \frac{p_3 + \theta}{p_4 + \theta} \right)$$

| 거리 (m) | 상대 침투 깊이 |
|----------|----------------|
| 0 | 최대 (100%) |
| 50 | ~90% |
| 100 | ~75% |
| 200 | ~60% |

---

## 9. 코드 구현 파일

| 파일 | 클래스/함수 | 설명 |
|------|-------------|------|
| `src/core/physics.py` | `CRNPPhysics.calculate_penetration_depth()` | 침투 깊이 $D$ 계산 |
| `src/core/physics.py` | `CRNPPhysics.depth_weighting_function()` | 깊이 가중 함수 $W_d$ |
| `src/core/physics.py` | `CRNPPhysics.calculate_attenuation_length()` | 감쇠 길이 $L$ 계산 |
| `src/core/footprint.py` | `FootprintCalculator.compute_depth_profile()` | 깊이 프로파일 및 D86 계산 |
| `src/core/analyzer.py` | `CRNPAnalyzer.analyze_single_day()` | 일별 분석 (깊이 포함) |

---

## 10. 출력 결과

### 10.1 `depth_profile` 딕셔너리 구조

```python
{
    'depths': np.array,           # 깊이 배열 (cm)
    'weights': np.array,          # 정규화된 가중치
    'cumulative': np.array,       # 누적 기여도
    'D': float,                   # 침투 깊이 (cm)
    'D86': float,                 # 86% 기여 깊이 (cm)
    'layer_contributions': [      # 레이어별 기여도
        {'layer': '0-10cm', 'contribution': 0.52, ...},
        {'layer': '10-20cm', 'contribution': 0.28, ...},
        ...
    ],
    'theta_used': float,          # 사용된 토양수분
    'bulk_density_used': float,   # 사용된 밀도
    'distance_r': float           # CRNP 거리
}
```

---

## 11. 참고 문헌

1. **Wang, T., et al. (2023).** Revisiting the CRNP measurement depth. *Vadose Zone Journal*, 22(4).

2. **Franz, T. E., et al. (2012).** Measurement depth of the cosmic ray soil moisture probe affected by hydrogen from various sources. *Water Resources Research*, 48(8).

3. **Köhli, M., et al. (2015).** Footprint characteristics revised for field-scale soil moisture monitoring with cosmic-ray neutrons. *Water Resources Research*, 51(7), 5772-5790.

4. **Zreda, M., et al. (2012).** COSMOS: The COsmic-ray Soil Moisture Observing System. *Hydrology and Earth System Sciences*, 16(11), 4079-4099.

---

## 12. 주요 개념 요약

| 용어 | 정의 | 주요 영향 인자 |
|------|------|----------------|
| **침투 깊이 (D)** | 중성자가 주로 감지되는 특성 깊이 | 토양수분, 밀도, 거리 |
| **D86** | 86% 신호가 유래하는 깊이 | D의 함수 |
| **깊이 가중 함수** | 깊이별 신호 기여도 | 지수 감쇠 형태 |
| **감쇠 길이 (L)** | 신호가 1/e로 감소하는 거리 | 토양수분 |

---

## 13. 실용적 고려사항

### 13.1 건조 vs 습윤 토양

- **건조 토양**: 유효깊이 깊음 (30-50 cm) → 더 많은 토양층 평균
- **습윤 토양**: 유효깊이 얕음 (10-20 cm) → 표층 위주 측정

### 13.2 토양 밀도의 영향

- **압밀된 토양** ($\rho_b$ 높음): 유효깊이 감소
- **느슨한 토양** ($\rho_b$ 낮음): 유효깊이 증가

### 13.3 계절 변화

토양수분의 계절적 변화에 따라 유효깊이도 동적으로 변화한다:
- 우기: 유효깊이 감소
- 건기: 유효깊이 증가
