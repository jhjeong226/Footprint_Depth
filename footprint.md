# CRNP 관측반경 (Footprint) 산정 이론 및 프로세스

## 1. 개요

우주선 중성자 탐지법(Cosmic-Ray Neutron Probe, CRNP)에서 **관측반경(Footprint)**은 중성자 센서가 토양수분을 측정하는 공간적 영역을 의미한다. 이 코드는 **Schrön et al. (2017, 2023)** 및 **Köhli et al. (2015)**의 이론을 기반으로 관측반경을 계산한다.

---

## 2. 수평 가중 함수 (Radial Weighting Function)

### 2.1 기본 수식

수평 가중 함수 $W_r(r, h, \theta)$는 CRNP로부터 거리 $r$에 위치한 토양이 중성자 신호에 기여하는 정도를 나타낸다.

**수식 (Schrön et al. 2023, Equation 2):**

$$W_r(r) = \left( a_1 \cdot e^{-r/b_1} + a_2 \cdot e^{-r/b_2} \right) \cdot \left( 1 - e^{-c \cdot r} \right)$$

### 2.2 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| $a_1$ | 30.0 | 근거리 가중치 (near-field weight) |
| $b_1$ | 1.6 m | 근거리 감쇠 스케일 (near-field decay scale) |
| $a_2$ | 1.0 | 원거리 가중치 (far-field weight) |
| $b_2$ | 100.0 m | 원거리 감쇠 스케일 (far-field decay scale) |
| $c$ | 3.7 | 근거리 억제 계수 (near-field suppression) |

### 2.3 수식의 물리적 의미

1. **이중 지수 감쇠**: $a_1 \cdot e^{-r/b_1} + a_2 \cdot e^{-r/b_2}$
   - 근거리(~수 미터): 빠른 감쇠 ($b_1 = 1.6$ m)
   - 원거리(~수백 미터): 느린 감쇠 ($b_2 = 100$ m)

2. **근거리 억제**: $(1 - e^{-c \cdot r})$
   - $r \to 0$일 때 $W_r \to 0$ (센서 바로 아래는 기여도 낮음)
   - 중성자가 토양과 대기 사이를 이동하는 데 필요한 최소 거리 반영

---

## 3. 토양수분 및 대기습도 의존성

### 3.1 스케일링 팩터

토양수분($\theta$)과 대기 절대습도($h$)에 따라 감쇠 스케일이 조정된다.

**스케일링 수식:**

$$f_{scale} = f_\theta \cdot f_h$$

여기서:
- $f_\theta = 0.55 + 1.10 \cdot e^{-3.5 \cdot \theta}$ (토양수분 의존성)
- $f_h = 1.0 - 0.005 \cdot (h - h_{ref})$ (습도 의존성, $h_{ref} = 5$ g/m³)

**조정된 감쇠 스케일:**
- $b_1' = b_1 \cdot f_{scale}$
- $b_2' = b_2 \cdot f_{scale}$

### 3.2 물리적 해석

| 조건 | $f_{scale}$ | Footprint | 설명 |
|------|-------------|-----------|------|
| 건조 토양 ($\theta$ 낮음) | > 1 | 증가 | 중성자가 멀리까지 이동 |
| 습윤 토양 ($\theta$ 높음) | < 1 | 감소 | 수소에 의한 감쇠 증가 |
| 고습도 대기 ($h$ 높음) | < 1 | 감소 | 대기 중 수소 증가 |

### 3.3 문헌값과의 비교 (Schrön et al. 2023, Table 1)

| 토양수분 ($\theta$) | R86 (관측반경) |
|---------------------|----------------|
| 5% | ~218 m |
| 20% | ~170 m |
| 30% | ~137 m |
| 40% | ~121 m |

---

## 4. R86 계산 (86% 누적 기여 반경)

### 4.1 정의

$R_{86}$은 전체 중성자 신호의 86%가 유래하는 원형 영역의 반경이다.

**수식 (Schrön et al. 2023, Equation 4):**

$$0.86 \times N(h, \theta) = \int_0^{R_{86}} W_r(r, h, \theta) \, dr$$

### 4.2 계산 방법

#### 방법 1: 해석적 방법 (max_extent 독립적)

```python
def calculate_analytical_R86(theta, h=5.0, target=0.86, r_max=500):
    r = np.linspace(0.1, r_max, 1000)
    Wr = radial_intensity_function(r, h, theta)
    cumulative = np.cumsum(Wr) * dr
    cumulative_norm = cumulative / cumulative[-1]
    R86 = r[np.searchsorted(cumulative_norm, target)]
    return R86
```

#### 방법 2: 그리드 기반 방법

2D 그리드에서 기여도를 계산하고, 거리순으로 정렬하여 누적합이 86%에 도달하는 반경을 찾는다.

### 4.3 방향별 R86 계산

비등방성(anisotropic) footprint를 분석하기 위해 방위각 섹터별로 R86을 계산한다.

```python
def directional_Rp_by_sector(Xi, Yi, contribution, p=0.86, ddeg=30):
    # 섹터 분할 (0-360도, ddeg 간격)
    for phi0 in range(0, 360, ddeg):
        # 해당 섹터 내 기여도의 86%가 포함되는 반경 계산
```

---

## 5. 기압 보정

### 5.1 보정 수식

대기압 변화에 따른 중성자 플럭스 변화를 보정한다.

$$s_P = \left( \frac{P}{P_0} \right)^{\alpha_P}$$

여기서:
- $P$: 현재 기압 (hPa)
- $P_0$: 표준 기압 (1013.25 hPa)
- $\alpha_P$: 스케일링 지수 (기본값 1.0)

### 5.2 유효 거리

$$R_{eff} = R \cdot s_P$$

고도가 높은 지역(기압 낮음)에서는 $s_P < 1$이므로 유효 거리가 감소한다.

---

## 6. 식생 보정

### 6.1 보정 수식 (Schrön et al. 2017)

식생이 있으면 중성자 신호가 감쇠된다.

$$f_{veg} = 1 - c_1 \cdot (1 - e^{-c_2 \cdot H_{veg}}) \cdot (1 + e^{-c_3 \cdot \theta})$$

### 6.2 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| $c_1$ | 0.17 | 최대 감쇠 비율 |
| $c_2$ | 0.41 | 식생 높이 스케일 (1/m) |
| $c_3$ | 7.0 | 토양수분 의존성 |
| $H_{veg}$ | - | 식생 높이 (m) |

---

## 7. 최종 기여도 계산

### 7.1 통합 수식

각 그리드 셀의 기여도 $C$:

$$C(x, y) = \frac{W_r(R_{eff}) \cdot f_{veg}(\theta, H_{veg}) \cdot N(\theta)}{R_{eff} + 1}$$

여기서:
- $N(\theta)$: 중성자-토양수분 관계 함수 (Köhli et al. 2021)

### 7.2 정규화

전체 기여도의 합이 1이 되도록 정규화:

$$C_{norm}(x, y) = \frac{C(x, y)}{\sum_{all} C}$$

---

## 8. 중성자-토양수분 관계

### 8.1 Köhli et al. (2021) 모델

$$N(\theta, h) = N_0 \cdot \frac{p_1 + p_2 \cdot \theta}{p_1 + \theta} \cdot \left( p_3 + p_4 \cdot h + p_5 \cdot h^2 + e^{-p_6 \cdot \theta} \cdot (p_7 + p_8 \cdot h) \right)$$

### 8.2 파라미터 (HC 사이트 예시)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| $N_0$ | 1753 | 건조 토양 중성자 카운트 (사이트별 고유값) |
| $p_1$ | 0.0226 | |
| $p_2$ | 0.207 | |
| $p_3$ | 1.024 | |
| $p_4$ | -0.0093 | |
| $p_5$ | 0.000074 | |
| $p_6$ | 1.625 | |
| $p_7$ | 0.235 | |
| $p_8$ | -0.0029 | |

---

## 9. 코드 구현 파일

| 파일 | 클래스/함수 | 설명 |
|------|-------------|------|
| `src/core/physics.py` | `CRNPPhysics.radial_intensity_function()` | 수평 가중 함수 |
| `src/core/physics.py` | `CRNPPhysics.calculate_analytical_R86()` | 해석적 R86 계산 |
| `src/core/physics.py` | `CRNPPhysics.vegetation_correction_factor()` | 식생 보정 |
| `src/core/footprint.py` | `FootprintCalculator.calculate_R86_radial()` | 그리드 기반 R86 |
| `src/core/footprint.py` | `FootprintCalculator.directional_Rp_by_sector()` | 방향별 R86 |
| `src/core/analyzer.py` | `CRNPAnalyzer.compute_contribution()` | 최종 기여도 계산 |

---

## 10. 참고 문헌

1. **Schrön, M., et al. (2017).** Improving calibration and validation of cosmic-ray neutron sensors in the light of spatial sensitivity. *Hydrology and Earth System Sciences*, 21(10), 5009-5030.

2. **Schrön, M., et al. (2023).** Cosmic-ray neutron rover surveys of field soil moisture and the influence of roads. *Water Resources Research*, 59(6).

3. **Köhli, M., et al. (2015).** Footprint characteristics revised for field‐scale soil moisture monitoring with cosmic‐ray neutrons. *Water Resources Research*, 51(7), 5772-5790.

4. **Köhli, M., et al. (2021).** Soil moisture and air humidity dependence of the above-ground cosmic-ray neutron intensity. *Frontiers in Water*, 2, 544847.

5. **Desilets, D., et al. (2010).** Nature's neutron probe: Land surface hydrology at an elusive scale with cosmic rays. *Water Resources Research*, 46(11).
