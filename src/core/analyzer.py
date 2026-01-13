"""
CRNP 분석 통합 클래스
"""
import os
import pandas as pd
import numpy as np
from .physics import CRNPPhysics
from .interpolation import SWCInterpolator
from .footprint import FootprintCalculator

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
        
        # Pressure 데이터 로드 (optional)
        self.pressure_daily = None
        if pressure_path is not None and os.path.exists(pressure_path):
            self.pressure_daily = self.load_pressure_daily(pressure_path)
        
        print(f"✓ Loaded {len(self.sensor_ids)} sensors")
        print(f"✓ Date range: {self.swc_df['Date'].min()} to {self.swc_df['Date'].max()}")
        print(f"✓ CRNP center: ({self.crnp_lat:.6f}, {self.crnp_lon:.6f})")
        if self.pressure_daily is not None:
            print(f"✓ Pressure series loaded: {len(self.pressure_daily)} daily values")
    
    def calculate_relative_coordinates(self):
        """CRNP 중심 기준 상대 좌표 계산"""
        lat_to_m = 111000.0
        lon_to_m = 111000.0 * np.cos(np.radians(self.crnp_lat))
        
        self.geo_df["x"] = (self.geo_df["lon"] - self.crnp_lon) * lon_to_m
        self.geo_df["y"] = (self.geo_df["lat"] - self.crnp_lat) * lat_to_m
    
    def get_sensor_locations(self):
        """센서 위치 반환"""
        return self.geo_df[["x", "y"]].to_numpy(dtype=float)
    
    def load_pressure_daily(self, pressure_path):
        """일별 기압 데이터 로드"""
        df = pd.read_excel(pressure_path, header=1)
        
        if "TIMESTAMP" not in df.columns:
            raise ValueError("Pressure file must have 'TIMESTAMP' column.")
        
        if "Air_Press_Avg" not in df.columns:
            if df.shape[1] < 5:
                raise ValueError("Cannot locate pressure column.")
            pressure_col = df.columns[4]
        else:
            pressure_col = "Air_Press_Avg"
        
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["TIMESTAMP"]).copy()
        
        df[pressure_col] = pd.to_numeric(df[pressure_col], errors="coerce")
        df = df.dropna(subset=[pressure_col]).copy()
        
        df["date"] = df["TIMESTAMP"].dt.normalize()
        daily = df.groupby("date")[pressure_col].mean().sort_index()
        daily.name = "pressure_hpa"
        return daily
    
    def get_daily_pressure(self, date_str):
        """특정 날짜 기압 가져오기"""
        if self.pressure_daily is None:
            return None
        d = pd.to_datetime(date_str).normalize()
        if d in self.pressure_daily.index:
            return float(self.pressure_daily.loc[d])
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
                            P0=1013.25, alpha_p=1.0):
        """
        Footprint (contribution) 계산
        
        Returns:
            C: Contribution map
            sP: Pressure scale factor
            K: Normalized kernel (순수 커널)
        """
        R = np.sqrt(Xi**2 + Yi**2)
        mask_r = R <= max_extent
        
        # Pressure scaling
        if (not use_pressure) or (pressure_hpa is None) or (not np.isfinite(pressure_hpa)):
            sP = 1.0
        else:
            sP = (float(pressure_hpa) / float(P0)) ** float(alpha_p)
        
        R_eff = R * sP
        
        theta = swc_map
        Hveg = float(height)
        
        # 수평 가중 함수
        Wr = self.physics.radial_intensity_function(R_eff, theta=theta)
        
        # 순수 커널 (vegetation/soil moisture 제외)
        K = np.zeros_like(R, dtype=float)
        mK = mask_r & np.isfinite(Wr) & np.isfinite(R_eff)
        K[mK] = Wr[mK] / (R_eff[mK] + 1.0)
        Ks = K.sum()
        if Ks > 0:
            K /= Ks
        
        # 식생 보정
        fveg = self.physics.vegetation_correction_factor(theta, Hveg)
        
        # 중성자-토양수분 관계
        N0 = self.config.physics['neutron_moisture']['N0']
        params = self.config.physics['neutron_moisture']
        p1, p2 = params['p1'], params['p2']
        N_theta = N0 * (p1 + p2 * theta) / (p1 + theta)
        
        # 최종 기여도
        mask = mask_r & np.isfinite(theta) & np.isfinite(N_theta)
        C = np.zeros_like(R, dtype=float)
        C[mask] = Wr[mask] * fveg[mask] * N_theta[mask] / (R_eff[mask] + 1.0)
        
        s = C.sum()
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
        
        # Contribution 계산
        contribution, sP, kernel_norm = self.compute_contribution(
            Xi, Yi, swc_map, veg_height,
            max_extent=max_extent,
            pressure_hpa=pressure_hpa,
            use_pressure=use_pressure,
            P0=P0,
            alpha_p=alpha_p
        )
        
        # Predicted SWC
        predicted_swc = float(np.nansum(contribution * swc_map))
        
        # R86 계산
        R86_radius, R86_cum = self.footprint_calc.calculate_R86_radial(
            Xi, Yi, contribution, target=0.86
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
        
        return {
            "date": date_str,
            "doy": doy,
            "predicted_swc": predicted_swc,
            "mean_swc": mean_swc,
            "std_swc": std_swc,
            "veg_height": veg_height,
            "pressure_hpa": pressure_hpa,
            "pressure_scale_sP": sP,
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