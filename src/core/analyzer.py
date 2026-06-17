"""
CRNP 분석 통합 클래스
"""
import os
import pandas as pd
import numpy as np
from .physics import CRNPPhysics
from .interpolation import SWCInterpolator
from .footprint import FootprintCalculator
from .signal_contribution import SignalContribution

class CRNPAnalyzer:
    """CRNP Footprint 분석기 (Horizontal + Vertical)"""
    
    def __init__(self, config, geo_path, swc_path, pressure_path=None):
        """
        Args:
            config: Config 객체
            geo_path: 지리 정보 파일 경로
            swc_path: SWC 데이터 파일 경로
            pressure_path: 기압 데이터 파일 경로 (optional)
        """
        self.config = config
        
        # 물리 계산 초기화
        self.physics = CRNPPhysics(config)
        
        # SWC 보간 초기화 - Config 객체 전체 전달
        self.interpolator = SWCInterpolator(config)
        
        # Footprint 계산기
        self.footprint_calc = FootprintCalculator(config)

        # 신호 기여도 (원거리 기여도 / 실용적 footprint)
        self.sigcon = SignalContribution(self.physics, config)

        # 데이터 로드
        self._load_data(geo_path, swc_path, pressure_path)
        
    def _load_data(self, geo_path, swc_path, pressure_path=None):
        """
        데이터 파일 로드
        
        Args:
            geo_path: 지리 정보 파일 경로
            swc_path: SWC 데이터 파일 경로  
            pressure_path: 기압 데이터 파일 경로 (optional)
        """
        import os
        import pandas as pd
        
        # GEO 파일 로드
        geo_df_raw = pd.read_excel(geo_path)
        
        # CRNP 위치 찾기
        if "id" in geo_df_raw.columns:
            crnp_row = geo_df_raw[geo_df_raw["id"] == "CRNP"]
        else:
            crnp_row = pd.DataFrame()
        
        if len(crnp_row) > 0:
            self.crnp_lat = float(crnp_row["lat"].values[0])
            self.crnp_lon = float(crnp_row["lon"].values[0])
        else:
            self.crnp_lat = float(geo_df_raw["lat"].mean())
            self.crnp_lon = float(geo_df_raw["lon"].mean())
        
        # 센서만 필터링
        if "id" in geo_df_raw.columns:
            self.geo_df = geo_df_raw[geo_df_raw["id"] != "CRNP"].copy()
        else:
            self.geo_df = geo_df_raw.copy()
        
        # 상대 좌표 계산
        self.calculate_relative_coordinates()
        
        # SWC 파일 로드
        self.swc_df = pd.read_excel(swc_path)
        
        if "Date" not in self.swc_df.columns:
            raise ValueError("SWC file must contain a 'Date' column.")
        
        self.swc_df["Date"] = pd.to_datetime(self.swc_df["Date"], errors="coerce")
        
        # 센서 ID 목록
        self.sensor_ids = [c for c in self.swc_df.columns if c != "Date"]
        if len(self.sensor_ids) == 0:
            raise ValueError("No sensor columns found in SWC file.")
        
        # Pressure 및 Humidity 데이터 로드 (optional)
        self.pressure_daily = None
        self.humidity_daily = None
        if pressure_path is not None and os.path.exists(pressure_path):
            self.pressure_daily, self.humidity_daily = self.load_climate_daily(pressure_path)

        print(f"[OK] Loaded {len(self.sensor_ids)} sensors")
        print(f"[OK] Date range: {self.swc_df['Date'].min()} to {self.swc_df['Date'].max()}")
        print(f"[OK] CRNP center: ({self.crnp_lat:.6f}, {self.crnp_lon:.6f})")
        if self.pressure_daily is not None:
            print(f"[OK] Climate series loaded: {len(self.pressure_daily)} daily values")
            print(f"  - Pressure (hPa)")
            if self.humidity_daily is not None:
                print(f"  - Absolute humidity (g/m3)")
    
    def calculate_relative_coordinates(self):
        """CRNP 중심 기준 상대 좌표 계산"""
        lat_to_m = 111000.0
        lon_to_m = 111000.0 * np.cos(np.radians(self.crnp_lat))
        
        self.geo_df["x"] = (self.geo_df["lon"] - self.crnp_lon) * lon_to_m
        self.geo_df["y"] = (self.geo_df["lat"] - self.crnp_lat) * lat_to_m
    
    def get_sensor_locations(self):
        """센서 위치 반환"""
        return self.geo_df[["x", "y"]].to_numpy(dtype=float)
    
    @staticmethod
    def calculate_absolute_humidity(temp_c, rh_percent):
        """
        절대습도 계산 (g/m³)

        Args:
            temp_c: 온도 (°C)
            rh_percent: 상대습도 (%)

        Returns:
            절대습도 (g/m³)
        """
        # Magnus 공식으로 포화수증기압 계산 (hPa)
        es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

        # 실제 수증기압 (hPa)
        e = es * (rh_percent / 100.0)

        # 절대습도 (g/m³)
        # AH = (e × 100) / ((T + 273.15) × 0.4615)
        # 여기서 e는 hPa, T는 °C
        abs_humidity = (e * 100.0) / ((temp_c + 273.15) * 0.4615)

        return abs_humidity

    def load_climate_daily(self, climate_path):
        """
        일별 기후 데이터 로드 (기압, 절대습도)

        Args:
            climate_path: 기후 데이터 파일 경로

        Returns:
            pressure_daily: 일별 기압 (hPa)
            humidity_daily: 일별 절대습도 (g/m³)
        """
        df = pd.read_excel(climate_path, header=1)

        if "TIMESTAMP" not in df.columns:
            raise ValueError("Climate file must have 'TIMESTAMP' column.")

        # 기압 컬럼 찾기
        if "Air_Press_Avg" not in df.columns:
            if df.shape[1] < 5:
                raise ValueError("Cannot locate pressure column.")
            pressure_col = df.columns[4]
        else:
            pressure_col = "Air_Press_Avg"

        # 온도 및 습도 컬럼 확인
        temp_col = "Air_Temp_Avg" if "Air_Temp_Avg" in df.columns else None
        rh_col = "RH_Avg" if "RH_Avg" in df.columns else None

        # 타임스탬프 처리
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["TIMESTAMP"]).copy()

        # 기압 데이터 처리
        df[pressure_col] = pd.to_numeric(df[pressure_col], errors="coerce")

        # 절대습도 계산 (온도와 상대습도가 있을 경우)
        humidity_daily = None
        if temp_col and rh_col:
            df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
            df[rh_col] = pd.to_numeric(df[rh_col], errors="coerce")

            # 절대습도 계산
            valid_mask = df[temp_col].notna() & df[rh_col].notna()
            df.loc[valid_mask, "abs_humidity"] = df.loc[valid_mask].apply(
                lambda row: self.calculate_absolute_humidity(row[temp_col], row[rh_col]),
                axis=1
            )

            # 일별 평균
            df["date"] = df["TIMESTAMP"].dt.normalize()
            humidity_daily = df.groupby("date")["abs_humidity"].mean().sort_index()
            humidity_daily.name = "abs_humidity_gm3"

        # 기압 일별 평균
        df["date"] = df["TIMESTAMP"].dt.normalize()
        pressure_daily = df.groupby("date")[pressure_col].mean().sort_index()
        pressure_daily.name = "pressure_hpa"

        return pressure_daily, humidity_daily
    
    def get_daily_pressure(self, date_str):
        """특정 날짜 기압 가져오기"""
        if self.pressure_daily is None:
            return None
        d = pd.to_datetime(date_str).normalize()
        if d in self.pressure_daily.index:
            return float(self.pressure_daily.loc[d])
        return None

    def get_daily_humidity(self, date_str):
        """특정 날짜 절대습도 가져오기"""
        if self.humidity_daily is None:
            return None
        d = pd.to_datetime(date_str).normalize()
        if d in self.humidity_daily.index:
            return float(self.humidity_daily.loc[d])
        return None
    
    def get_daily_swc(self, date_str):
        """특정 날짜 SWC 가져오기"""
        target = pd.to_datetime(date_str).normalize()
        row = self.swc_df.loc[self.swc_df["Date"].dt.normalize() == target]
        if len(row) == 0:
            raise ValueError(f"No data for {date_str}")
        
        swc_values = row[self.sensor_ids].iloc[0].to_numpy(dtype=float)
        valid_mask = ~np.isnan(swc_values)
        return swc_values, valid_mask
    
    def estimate_vegetation_height(self, doy, base_height=0.3):
        """DOY 기반 식생 높이 추정"""
        growth_start, growth_end, peak_doy = 120, 300, 210
        if doy < growth_start or doy > growth_end:
            return 0.05
        
        if doy <= peak_doy:
            progress = (doy - growth_start) / (peak_doy - growth_start)
        else:
            progress = 1 - (doy - peak_doy) / (growth_end - peak_doy)
        
        h = base_height * np.sin(progress * np.pi / 2)
        return max(0.05, float(h))
    
    def compute_contribution(self, Xi, Yi, swc_map, height, max_extent=170,
                            pressure_hpa=None, use_pressure=True,
                            P0=1013.25, alpha_p=1.0, abs_humidity=None,
                            bulk_density=1.4):
        """
        Footprint (contribution) 계산

        Args:
            Xi, Yi: 그리드 좌표
            swc_map: 토양수분 맵
            height: 식생 높이
            max_extent: 최대 범위
            pressure_hpa: 기압 (hPa)
            use_pressure: 기압 보정 사용 여부
            P0: 표준 기압
            alpha_p: 기압 스케일링 지수
            abs_humidity: 절대습도 (g/m³)
            bulk_density: 토양 밀도 (g/cm³)

        Returns:
            C: Contribution map
            sP: Pressure scale factor
            K: Normalized kernel (순수 커널)
        """
        R = np.sqrt(Xi**2 + Yi**2)
        mask_r = R <= max_extent

        theta = swc_map
        Hveg = float(height)
        h = abs_humidity if abs_humidity is not None else 5.0

        # 대표 토양수분 (r* 재스케일링 파라미터 함수는 스칼라 조건 가정)
        theta_repr = float(np.nanmean(theta[mask_r]))

        # r* 재스케일링: 기압 + 식생 + 전제 토양수분 (Schrön 2017)
        R_star = self.physics.rescale_distance(
            R, pressure_hpa=pressure_hpa if use_pressure else None,
            Hveg=Hveg, theta=theta_repr, P0=P0
        )

        # 기압 스케일 팩터 (보고용)
        if (not use_pressure) or (pressure_hpa is None) or (not np.isfinite(float(pressure_hpa) if pressure_hpa is not None else float('nan'))):
            sP = 1.0
        else:
            sP = 0.4922 / (0.86 - np.exp(-float(pressure_hpa) / float(P0)))

        # 완전한 Wr (r* 입력)
        Wr = self.physics.radial_intensity_function(R_star, h=h, theta=theta_repr, bulk_density=bulk_density)

        # 순수 커널 K = Wr/r (공간 가중만, 정규화)
        K = np.zeros_like(R, dtype=float)
        mK = mask_r & np.isfinite(Wr) & (R > 0)
        K[mK] = Wr[mK] / R[mK]
        Ks = np.nansum(K)
        if Ks > 0:
            K /= Ks

        # 식생 보정 및 중성자-토양수분 관계
        fveg = self.physics.vegetation_correction_factor(theta, Hveg)
        N0 = self.config.physics['neutron_moisture']['N0']
        N_theta = self.physics.neutron_moisture_relationship(theta, h=h, N0=N0)

        # 최종 기여도 C = Wr/r × fveg × N(θ)
        mask = mask_r & np.isfinite(theta) & np.isfinite(N_theta) & (R > 0)
        C = np.zeros_like(R, dtype=float)
        C[mask] = Wr[mask] * fveg[mask] * N_theta[mask] / R[mask]

        s = np.nansum(C)
        if s > 0:
            C /= s

        return C, float(sP), K
    
    def analyze_single_day(self, date_str, base_veg_height=0.3,
                          max_extent=None, resolution=None, ddeg=30,
                          use_pressure=True, P0=1013.25, alpha_p=1.0,
                          bulk_density=None,
                          rbf_function=None,
                          rbf_smooth=None):
        """
        단일 날짜 분석
        
        Args:
            date_str: 날짜 문자열
            base_veg_height: 기본 식생 높이
            max_extent: 최대 범위 (None이면 config에서)
            resolution: 그리드 해상도 (None이면 config에서)
            ddeg: 섹터 각도
            use_pressure: 기압 보정 사용 여부
            P0: 표준 기압
            alpha_p: 기압 스케일링 지수
            bulk_density: 토양 밀도 (None이면 config에서)
        
        Returns:
            results: 분석 결과 딕셔너리
        """
        # Config에서 기본값 가져오기
        if max_extent is None:
            max_extent = self.config.analysis['max_extent']
        if resolution is None:
            resolution = self.config.analysis['resolution']
        if bulk_density is None:
            bulk_density = self.config.site['bulk_density']
        
        # SWC 데이터
        swc_values, valid_mask = self.get_daily_swc(date_str)
        
        # SWC 내삽
        Xi, Yi, swc_map = self.interpolator.interpolate(
            self.get_sensor_locations(),
            swc_values,
            valid_mask,
            max_extent=max_extent,
            resolution=resolution,
            rbf_function=rbf_function,
            rbf_smooth=rbf_smooth 
        )
        
        # 식생 높이
        date = pd.to_datetime(date_str)
        doy = date.timetuple().tm_yday
        veg_height = self.estimate_vegetation_height(doy, base_veg_height)

        # 기압
        pressure_hpa = self.get_daily_pressure(date_str) if use_pressure else None

        # 절대습도
        abs_humidity = self.get_daily_humidity(date_str)

        # Contribution 계산
        contribution, sP, kernel_norm = self.compute_contribution(
            Xi, Yi, swc_map, veg_height,
            max_extent=max_extent,
            pressure_hpa=pressure_hpa,
            use_pressure=use_pressure,
            P0=P0,
            alpha_p=alpha_p,
            abs_humidity=abs_humidity,
            bulk_density=bulk_density
        )
        
        # Predicted SWC
        predicted_swc = float(np.nansum(contribution * swc_map))
        
        # R86 계산 (그리드 기반 - max_extent 의존)
        R86_radius, R86_cum = self.footprint_calc.calculate_R86_radial(
            Xi, Yi, contribution, target=0.86
        )

        # 해석적 R86 계산 (physics.py 기반, 토양수분 반영)
        h_for_R86 = abs_humidity if abs_humidity is not None else 5.0
        R86_analytical = self.physics.calculate_analytical_R86(
            theta=predicted_swc,
            bulk_density=bulk_density,
            h=h_for_R86,
            pressure_hpa=pressure_hpa,
            Hveg=veg_height
        )

        # 방향별 R86
        sectors = self.footprint_calc.directional_Rp_by_sector(
            Xi, Yi, contribution, p=0.86, ddeg=ddeg, min_sector_mass=1e-4
        )
        
        # 통계
        mean_swc = float(np.nanmean(swc_values[valid_mask]))
        std_swc = float(np.nanstd(swc_values[valid_mask]))
        
        # 깊이 프로파일
        depth_profile = self.footprint_calc.compute_depth_profile(
            theta=predicted_swc,
            bulk_density=bulk_density,
            r=0,
            max_depth=100,
            n_points=100
        )
        
        # 실용적 footprint 거리 (Schrön 2023 Eq.14)
        press_for_fp = pressure_hpa if pressure_hpa is not None else 1013.25
        practical_fp_wet = self.sigcon.practical_footprint_distance(
            predicted_swc, dtheta=+0.10, sigma_N=0.01,
            h=h_for_R86, pressure_hpa=press_for_fp, Hveg=veg_height)
        practical_fp_dry = self.sigcon.practical_footprint_distance(
            predicted_swc, dtheta=-0.10, sigma_N=0.01,
            h=h_for_R86, pressure_hpa=press_for_fp, Hveg=veg_height)

        return {
            "date": date_str,
            "doy": doy,
            "predicted_swc": predicted_swc,
            "mean_swc": mean_swc,
            "std_swc": std_swc,
            "veg_height": veg_height,
            "pressure_hpa": pressure_hpa,
            "pressure_scale_sP": sP,
            "abs_humidity": abs_humidity,
            "Xi": Xi,
            "Yi": Yi,
            "swc_map": swc_map,
            "contribution": contribution,
            "max_extent": max_extent,
            "resolution": resolution,
            "swc_values": swc_values,
            "valid_mask": valid_mask,
            "n_valid_sensors": int(np.sum(valid_mask)),
            "R86_radius": R86_radius,
            "R86_cum": R86_cum,
            "R86_analytical": R86_analytical,
            "practical_footprint_wet_m": practical_fp_wet,
            "practical_footprint_dry_m": practical_fp_dry,
            "ddeg": ddeg,
            "R86_phi_sectors": sectors,
            "use_pressure": use_pressure,
            "P0": float(P0),
            "alpha_p": float(alpha_p),
            "depth_profile": depth_profile,
            "bulk_density": bulk_density,
            "kernel_norm": kernel_norm,
            "sensor_locations": self.get_sensor_locations(),
            "rbf_function": rbf_function or self.config.interpolation.get('rbf_function', 'thin_plate'),
            "rbf_smooth": rbf_smooth or self.config.interpolation.get('rbf_smooth', 0.0),
        }