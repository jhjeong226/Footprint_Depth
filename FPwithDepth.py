"""
CRNP Footprint Analysis with Depth Profile (Complete Integrated Version)
=========================================================================

Fix (2026-01-09):
- The red boundary line is NOT a kernel contour.
- To avoid "lying" visualization in uniform theta case:
  * Always compute kernel-based radial R86 -> draw as perfect circle (baseline)
  * Draw directional R86(phi) polyline ONLY when redistribution is real (L1(C-K) > tol)
  * Do NOT overlay C-based amoeba on the kernel panel

Author: Integrated version with depth analysis and cross-section view
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, TwoSlopeNorm
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.size"] = 10
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Utility Functions
# ============================================================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _base_axes_setup(ax, max_extent, title, xlabel="Distance (m)", ylabel="Distance (m)"):
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

def robust_norm_for_contribution(C, mode="log"):
    """Contribution 분포 시각화를 위한 norm 설정"""
    C = np.asarray(C)
    Cp = C[C > 0]
    if Cp.size == 0:
        return None, 0.0

    eps = float(np.min(Cp)) * 0.5
    vmin = float(np.percentile(Cp, 5.0))
    vmax = float(np.percentile(C, 99.5))

    vmin = max(vmin, eps)
    vmax = max(vmax, vmin * 10)

    if mode == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = PowerNorm(gamma=0.35, vmin=vmin, vmax=vmax)

    return norm, eps

def build_amoeba_boundary_polyline(sectors, arc_step_deg=1.0):
    """
    30° 섹터별 rp로 연속 경계선 생성 (polyline)
    IMPORTANT: this is NOT a contour line.
    """
    if not sectors:
        return np.array([]), np.array([])

    sectors = sorted(sectors, key=lambda s: s["phi0"])
    xs, ys = [], []
    n = len(sectors)

    for i in range(n):
        s = sectors[i]
        phi0, phi1, rp = s["phi0"], s["phi1"], s["rp"]

        phis = np.arange(phi0, phi1 + arc_step_deg, arc_step_deg)
        phir = np.deg2rad(phis)
        xs.extend((rp * np.cos(phir)).tolist())
        ys.extend((rp * np.sin(phir)).tolist())

        # connect to next sector radius at the sector boundary angle
        s_next = sectors[(i + 1) % n]
        rp_next = s_next["rp"]
        phib = np.deg2rad(phi1 % 360.0)

        xs.append(rp * np.cos(phib))
        ys.append(rp * np.sin(phib))
        xs.append(rp_next * np.cos(phib))
        ys.append(rp_next * np.sin(phib))

    xs.append(xs[0])
    ys.append(ys[0])
    return np.array(xs), np.array(ys)

def plot_sector_rays(ax, max_extent, ddeg, **plot_kwargs):
    """섹터 경계 레이 표시"""
    for ang in np.arange(0, 360, ddeg):
        rad = np.deg2rad(ang)
        ax.plot([0, max_extent*np.cos(rad)], [0, max_extent*np.sin(rad)], **plot_kwargs)

def plot_circle(ax, radius, color="white", lw=2.0, alpha=1.0, zorder=6):
    """완전 원형 경계"""
    if (radius is None) or (not np.isfinite(radius)):
        return
    ang = np.linspace(0, 2*np.pi, 720)
    ax.plot(radius*np.cos(ang), radius*np.sin(ang), color=color, lw=lw, alpha=alpha, zorder=zorder)

def _convex_hull_mask(xy_points, Xi, Yi, buffer_m=0.0):
    """Convex hull 내부만 True 반환"""
    pts = np.asarray(xy_points, float)
    if pts.shape[0] < 3:
        return np.ones_like(Xi, dtype=bool)

    if buffer_m > 0:
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        scale = (r + buffer_m) / (r + 1e-12)
        pts = pts * scale[:, None]

    tri = Delaunay(pts)
    test = np.c_[Xi.ravel(), Yi.ravel()]
    inside = tri.find_simplex(test) >= 0
    return inside.reshape(Xi.shape)


# ============================================================
# Main Analyzer Class
# ============================================================

class CRNPAnalyzer:
    """CRNP Footprint 분석기 (Horizontal + Vertical)"""

    def __init__(self, geo_path, swc_path, pressure_path=None):
        geo_df_raw = pd.read_excel(geo_path)
        self.swc_df = pd.read_excel(swc_path)

        if "Date" not in self.swc_df.columns:
            raise ValueError("SWC file must contain a 'Date' column.")

        self.swc_df["Date"] = pd.to_datetime(self.swc_df["Date"], errors="coerce")

        # CRNP 위치
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

        self.calculate_relative_coordinates()

        self.sensor_ids = [c for c in self.swc_df.columns if c != "Date"]
        if len(self.sensor_ids) == 0:
            raise ValueError("No sensor columns found.")

        # Pressure data
        self.pressure_daily = None
        if pressure_path is not None:
            self.pressure_daily = self.load_pressure_daily(pressure_path)

        print(f"✓ Loaded {len(self.sensor_ids)} sensors")
        print(f"✓ Date range: {self.swc_df['Date'].min()} to {self.swc_df['Date'].max()}")
        print(f"✓ CRNP center: ({self.crnp_lat:.6f}, {self.crnp_lon:.6f})")
        if self.pressure_daily is not None:
            print(f"✓ Pressure series loaded: {len(self.pressure_daily)} daily values")

    def calculate_relative_coordinates(self):
        lat_to_m = 111000.0
        lon_to_m = 111000.0 * np.cos(np.radians(self.crnp_lat))

        self.geo_df["x"] = (self.geo_df["lon"] - self.crnp_lon) * lon_to_m
        self.geo_df["y"] = (self.geo_df["lat"] - self.crnp_lat) * lat_to_m

    def get_sensor_locations(self):
        return self.geo_df[["x", "y"]].to_numpy(dtype=float)

    def load_pressure_daily(self, pressure_path):
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
        if self.pressure_daily is None:
            return None
        d = pd.to_datetime(date_str).normalize()
        if d in self.pressure_daily.index:
            return float(self.pressure_daily.loc[d])
        return None

    def get_daily_swc(self, date_str):
        target = pd.to_datetime(date_str).normalize()
        row = self.swc_df.loc[self.swc_df["Date"].dt.normalize() == target]
        if len(row) == 0:
            raise ValueError(f"No data for {date_str}")

        row = row.sort_values("Date")
        swc_values = row[self.sensor_ids].iloc[-1].to_numpy(dtype=float)
        valid_mask = ~np.isnan(swc_values)
        return swc_values, valid_mask

    def interpolate_swc(self, swc_values, valid_mask, resolution=5, method="rbf",
                       max_extent=150, edge_control=True, hull_buffer_m=15.0,
                       outside_fill="nan",
                       clip_min=0.01, clip_max=0.6,
                       fill_remaining_nans_with_nearest=False):
        """SWC 보간 with edge control"""
        loc = self.get_sensor_locations()
        x_valid = loc[valid_mask, 0]
        y_valid = loc[valid_mask, 1]
        swc_valid = swc_values[valid_mask]

        n_valid = len(swc_valid)
        if n_valid == 0:
            raise ValueError("No valid SWC sensors.")
        if n_valid <= 2 and method == "rbf":
            method = "nearest"

        xi = np.arange(-max_extent, max_extent + resolution, resolution)
        yi = np.arange(-max_extent, max_extent + resolution, resolution)
        Xi, Yi = np.meshgrid(xi, yi)

        if method == "rbf":
            rbf = Rbf(x_valid, y_valid, swc_valid, function="thin_plate", smooth=0)
            Zi = rbf(Xi, Yi)
        else:
            from scipy.interpolate import griddata
            Zi = griddata((x_valid, y_valid), swc_valid, (Xi, Yi), method=method)
            if np.isnan(Zi).any():
                Zi2 = griddata((x_valid, y_valid), swc_valid, (Xi, Yi), method="nearest")
                Zi = np.where(np.isnan(Zi), Zi2, Zi)

        # Edge control
        if edge_control and (n_valid >= 3):
            xy = np.c_[x_valid, y_valid]
            inside = _convex_hull_mask(xy, Xi, Yi, buffer_m=hull_buffer_m)

            if outside_fill == "nan":
                Zi = Zi.astype(float)
                Zi[~inside] = np.nan
            else:
                gx = Xi[..., None]
                gy = Yi[..., None]
                d2 = (gx - x_valid[None, None, :])**2 + (gy - y_valid[None, None, :])**2
                nn = np.argmin(d2, axis=2)
                Zi_nn = swc_valid[nn]
                Zi = np.where(inside, Zi, Zi_nn)

        if fill_remaining_nans_with_nearest and np.isnan(Zi).any():
            from scipy.interpolate import griddata
            Zi2 = griddata((x_valid, y_valid), swc_valid, (Xi, Yi), method="nearest")
            Zi = np.where(np.isnan(Zi), Zi2, Zi)

        Zi = np.where(np.isnan(Zi), np.nan, np.clip(Zi, clip_min, clip_max))
        return Xi, Yi, Zi

    def estimate_vegetation_height(self, doy, base_height=0.3):
        growth_start, growth_end, peak_doy = 120, 300, 210
        if doy < growth_start or doy > growth_end:
            return 0.05

        if doy <= peak_doy:
            progress = (doy - growth_start) / (peak_doy - growth_start)
        else:
            progress = 1 - (doy - peak_doy) / (growth_end - peak_doy)

        h = base_height * np.sin(progress * np.pi / 2)
        return max(0.05, float(h))

    def compute_footprint(self, Xi, Yi, swc_map, height, max_extent=150,
                         pressure_hpa=None, use_pressure=True, P0=1013.25, alpha_p=1.0):
        """
        Return:
        - C: theta/veg-modulated contribution normalized (sum=1)
        - sP: pressure scale factor
        - K: pure kernel normalized (sum=1), NO theta dependence
        """
        R = np.sqrt(Xi**2 + Yi**2)
        mask_r = R <= max_extent

        if (not use_pressure) or (pressure_hpa is None) or (not np.isfinite(pressure_hpa)):
            sP = 1.0
        else:
            sP = (float(pressure_hpa) / float(P0)) ** float(alpha_p)

        R_eff = R * sP

        theta = swc_map
        Hveg = float(height)

        Wr = (30 * np.exp(-R_eff / 1.6) + np.exp(-R_eff / 100.0)) * (1 - np.exp(-3.7 * R_eff))

        # --- Pure kernel (no theta)
        K = np.zeros_like(R, dtype=float)
        mK = mask_r & np.isfinite(Wr) & np.isfinite(R_eff)
        K[mK] = Wr[mK] / (R_eff[mK] + 1.0)
        Ks = np.nansum(K)
        if Ks > 0:
            K /= Ks

        # --- Theta-dependent modifier (still spatial if theta spatial)
        fveg = 1 - 0.17 * (1 - np.exp(-0.41 * Hveg)) * (1 + np.exp(-7 * theta))

        N0 = 1950.0
        p1, p2 = 0.0226, 0.207
        N_theta = N0 * (p1 + p2 * theta) / (p1 + theta)

        mask = mask_r & np.isfinite(theta) & np.isfinite(N_theta)

        C = np.zeros_like(R, dtype=float)
        C[mask] = Wr[mask] * fveg[mask] * N_theta[mask] / (R_eff[mask] + 1.0)

        s = np.nansum(C)
        if s > 0:
            C /= s

        return C, float(sP), K

    def calculate_R86_radial(self, Xi, Yi, contribution, target=0.86):
        R = np.sqrt(Xi**2 + Yi**2).ravel()
        C = contribution.ravel()
        order = np.argsort(R)
        R_sorted = R[order]
        C_sorted = C[order]
        cs = np.cumsum(C_sorted)
        idx = np.searchsorted(cs, target)
        if idx >= len(R_sorted):
            return np.nan, np.nan
        return float(R_sorted[idx]), float(cs[idx])

    def directional_Rp_by_sector(self, Xi, Yi, contribution, p=0.86, ddeg=15, min_sector_mass=1e-6):
        x = Xi.ravel()
        y = Yi.ravel()
        c = contribution.ravel()

        r = np.sqrt(x*x + y*y)
        phi = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

        sectors = []
        for phi0 in np.arange(0, 360, ddeg):
            phi1 = phi0 + ddeg
            m = (phi >= phi0) & (phi < phi1)
            if not np.any(m):
                continue

            rr = r[m]
            cc = c[m]
            sector_mass = np.sum(cc)
            if sector_mass < min_sector_mass:
                continue

            order = np.argsort(rr)
            rr = rr[order]
            cc = cc[order]
            cs = np.cumsum(cc)

            target = p * sector_mass
            k = np.searchsorted(cs, target)
            k = min(k, len(rr) - 1)

            rp = float(rr[k])
            sectors.append({"phi0": float(phi0), "phi1": float(phi1), "rp": rp})

        return sectors

    # ============================================================
    # Depth Analysis Methods (unchanged)
    # ============================================================

    def calculate_penetration_depth(self, theta, bulk_density=None, r=0):
        p0 = 8.321
        p1 = 0.14249
        p2 = 0.86655
        p3 = 26.42
        p4 = 0.0567

        if bulk_density is None:
            bulk_density = 1.5

        theta = np.asarray(theta)
        denominator = p4 + theta
        denominator = np.where(denominator == 0, 1e-10, denominator)

        term = (p3 + theta) / denominator
        D = (1.0 / bulk_density) * (p0 + p1 * (p2 + np.exp(-r/100.0) * term))

        return float(D) if np.isscalar(theta) else D

    def depth_weighting_function(self, depths, theta, D=None):
        depths = np.asarray(depths)
        if D is None:
            D = self.calculate_penetration_depth(theta)
        return np.exp(-2.0 * depths / D)

    def compute_depth_profile(self, theta, bulk_density=None, r=0, max_depth=100, n_points=100):
        depths = np.linspace(0, max_depth, n_points)
        D = self.calculate_penetration_depth(theta, bulk_density, r=r)

        weights = self.depth_weighting_function(depths, theta, D)
        weights_norm = weights / np.trapz(weights, depths)

        cumulative = np.array([np.trapz(weights_norm[:i+1], depths[:i+1])
                               for i in range(len(depths))])

        idx86 = np.searchsorted(cumulative, 0.86)
        idx86 = min(idx86, len(depths) - 1)
        D86 = float(depths[idx86])

        layers = [(0, 10), (10, 20), (20, 40), (40, 100)]
        layer_contrib = []
        for d_start, d_end in layers:
            idx_layer = (depths >= d_start) & (depths <= d_end)
            if np.any(idx_layer):
                contrib = np.trapz(weights_norm[idx_layer], depths[idx_layer])
                layer_contrib.append({
                    'layer': f'{d_start}-{d_end}cm',
                    'depth_start': d_start,
                    'depth_end': d_end,
                    'contribution': float(contrib),
                    'contribution_pct': float(contrib * 100)
                })

        return {
            'depths': depths,
            'weights': weights_norm,
            'cumulative': cumulative,
            'D': float(D),
            'D86': D86,
            'layer_contributions': layer_contrib,
            'theta_used': float(theta),
            'bulk_density_used': float(bulk_density) if bulk_density else 1.5,
            'distance_r': float(r)
        }

    # ============================================================
    # Single Day Analysis (with truthful boundary handling)
    # ============================================================

    def analyze_single_day(self, date_str, base_veg_height=0.3,
                           max_extent=150, resolution=10, ddeg=15,
                           use_pressure=True, P0=1013.25, alpha_p=1.0,
                           edge_control=True, hull_buffer_m=150.0, outside_fill="nearest",
                           bulk_density=None,
                           clip_min=0.05, clip_max=0.50,
                           fill_remaining_nans_with_nearest=True,
                           # ✅ 균일 판정 기준 (C와 K의 차이가 작으면 '원형만' 보여줌)
                           circle_l1_tol=3e-3):

        swc_values, valid_mask = self.get_daily_swc(date_str)

        Xi, Yi, swc_map = self.interpolate_swc(
            swc_values, valid_mask,
            resolution=resolution,
            method="rbf",
            max_extent=max_extent,
            edge_control=edge_control,
            hull_buffer_m=hull_buffer_m,
            outside_fill=outside_fill,
            clip_min=clip_min,
            clip_max=clip_max,
            fill_remaining_nans_with_nearest=fill_remaining_nans_with_nearest
        )

        date = pd.to_datetime(date_str)
        doy = date.timetuple().tm_yday
        veg_height = self.estimate_vegetation_height(doy, base_veg_height)

        pressure_hpa = self.get_daily_pressure(date_str) if use_pressure else None

        contribution, sP, kernel_norm = self.compute_footprint(
            Xi, Yi, swc_map, veg_height,
            max_extent=max_extent,
            pressure_hpa=pressure_hpa, use_pressure=use_pressure,
            P0=P0, alpha_p=alpha_p
        )

        predicted_swc = float(np.nansum(contribution * swc_map))

        # --- Radial R86 for contribution (C) and kernel (K)
        R86_C, _ = self.calculate_R86_radial(Xi, Yi, contribution, target=0.86)
        R86_K, _ = self.calculate_R86_radial(Xi, Yi, kernel_norm, target=0.86)

        # --- Redistribution magnitude (truth metric)
        l1_ck = float(np.nansum(np.abs(contribution - kernel_norm)))
        uniform_effective = bool(np.isfinite(l1_ck) and (l1_ck < circle_l1_tol))

        # --- Directional boundary from C (only meaningful when heterogeneity matters)
        sectors_C = self.directional_Rp_by_sector(
            Xi, Yi, contribution, p=0.86, ddeg=ddeg, min_sector_mass=1e-6
        )

        mean_swc = float(np.nanmean(swc_values[valid_mask]))
        std_swc = float(np.nanstd(swc_values[valid_mask]))

        depth_profile = self.compute_depth_profile(
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
            "kernel_norm": kernel_norm,
            "max_extent": max_extent,
            "resolution": resolution,
            "swc_values": swc_values,
            "valid_mask": valid_mask,
            "n_valid_sensors": int(np.sum(valid_mask)),
            # ✅ 저장: 경계 관련
            "R86_radius_C": R86_C,
            "R86_radius_K": R86_K,
            "R86_phi_sectors_C": sectors_C,
            "ddeg": ddeg,
            # ✅ 저장: '균일이면 원형만'을 위한 진실 지표
            "l1_ck": l1_ck,
            "uniform_effective": uniform_effective,
            "circle_l1_tol": float(circle_l1_tol),
            # etc
            "use_pressure": use_pressure,
            "P0": float(P0),
            "alpha_p": float(alpha_p),
            "edge_control": bool(edge_control),
            "hull_buffer_m": float(hull_buffer_m),
            "outside_fill": str(outside_fill),
            "depth_profile": depth_profile,
            "bulk_density": bulk_density,
            "clip_min": float(clip_min),
            "clip_max": float(clip_max),
        }


# ============================================================
# Diagnostics
# ============================================================

def print_footprint_diagnostics(results):
    K = results["kernel_norm"]
    C = results["contribution"]
    theta = results["swc_map"]

    eps = 1e-20
    mask = np.isfinite(K) & np.isfinite(C) & np.isfinite(theta) & (K > 0) & (C >= 0)

    logK = np.log10(K[mask] + eps)
    logC = np.log10(C[mask] + eps)

    corr = np.corrcoef(logK, logC)[0, 1]
    l1 = np.sum(np.abs(C[mask] - K[mask]))
    kl = np.sum(C[mask] * np.log((C[mask] + eps) / (K[mask] + eps)))

    print("\n" + "="*70)
    print(f"[DIAG] date = {results['date']}")
    print(f"[DIAG] corr(logK, logC) = {corr:.6f}")
    print(f"[DIAG] L1(C-K)          = {l1:.6e}  (tol={results['circle_l1_tol']:.2e})")
    print(f"[DIAG] KL(C||K)         = {kl:.6e}")
    print(f"[DIAG] uniform_effective = {results['uniform_effective']}")
    print("="*70 + "\n")


# ============================================================
# Plot Functions
# ============================================================

def mass_contour_field(W, eps=1e-30):
    """
    W(=kernel or contribution)가 주어졌을 때,
    각 픽셀 i에 대해:
      Q_i = sum(W_j where W_j >= W_i)
    를 계산한 필드 Q를 반환.
    즉, Q=0.86 등고선은 '상위 기여 영역이 86%가 되는 경계' (HDR mass contour).
    """
    W = np.asarray(W, float)
    W = np.where(np.isfinite(W), W, 0.0)
    W = np.clip(W, 0.0, None)

    flat = W.ravel()
    order = np.argsort(flat)[::-1]  # descending
    sorted_w = flat[order]

    csum = np.cumsum(sorted_w)
    total = csum[-1] if csum.size > 0 else 0.0
    if total <= eps:
        return np.zeros_like(W, dtype=float)

    csum /= total  # now [0,1]
    # rank-> cumulative mass of all cells with >= that value
    # for each original cell, we want Q_i = cumulative mass up to its rank in sorted order
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    Q_flat = csum[inv]
    return Q_flat.reshape(W.shape)

def draw_mass_contour(ax, Xi, Yi, W, level=0.86, color="red", lw=2.5, alpha=1.0, zorder=7):
    """
    W로부터 HDR mass contour (Q=level)을 계산해서 contour line을 그림.
    """
    Q = mass_contour_field(W)
    cs = ax.contour(Xi, Yi, Q, levels=[level], colors=[color], linewidths=lw, alpha=alpha)
    return cs

def plot_panel_swc(analyzer, results, save_path):
    max_extent = results["max_extent"]
    loc = analyzer.get_sensor_locations()
    swc_values = results["swc_values"]
    valid = results["valid_mask"]

    fig, ax = plt.subplots(figsize=(7.8, 6.5))

    Zi = results["swc_map"]
    Zi_masked = np.ma.masked_invalid(Zi)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(alpha=0.0)

    im = ax.imshow(
        Zi_masked,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap=cmap,
        vmin=0.15, vmax=0.40,
        interpolation="bilinear"
    )

    ax.scatter(
        loc[valid, 0], loc[valid, 1],
        c=swc_values[valid], s=160, marker="o",
        edgecolors="red", linewidths=1.4,
        cmap="Blues", vmin=0.15, vmax=0.40,
        zorder=5
    )

    if np.any(~valid):
        ax.scatter(
            loc[~valid, 0], loc[~valid, 1],
            s=160, marker="x", color="gray", linewidths=2,
            zorder=5
        )

    ax.plot(0, 0, "r+", markersize=18, markeredgewidth=3, zorder=6)

    title = f"Soil Moisture Distribution\n{results['date']}"
    title += f"\n(interp clip={results['clip_min']:.2f}-{results['clip_max']:.2f}, outside={results['outside_fill']})"

    _base_axes_setup(ax, max_extent, title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SWC (cm³/cm³)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_panel_veg(results, save_path):
    max_extent = results["max_extent"]
    fig, ax = plt.subplots(figsize=(7.8, 6.5))

    height_map = np.full_like(results["swc_map"], results["veg_height"], dtype=float)
    im = ax.imshow(
        height_map,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="YlGn",
        vmin=0, vmax=0.5,
        interpolation="bilinear"
    )
    ax.plot(0, 0, "r+", markersize=18, markeredgewidth=3, zorder=6)

    _base_axes_setup(ax, max_extent, f"Vegetation Height\n(DOY={results['doy']})")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (m)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_panel_footprint_diagnostics_3up(results, save_path,
                                        show_sector_rays=True,
                                        norm_mode="log",
                                        boundary_lw=2.5,
                                        mass_level=0.86):
    """
    Panel 3 (3-up) with TRUE mass-contours:
    (A) Pure kernel K + its 86% mass contour (should be circular)
    (B) Contribution C + its 86% mass contour (amoeba when hetero)
    (C) Delta (KT_share - K) + optional overlay of C contour
    """
    max_extent = results["max_extent"]
    ddeg = results["ddeg"]
    Xi, Yi = results["Xi"], results["Yi"]

    K = results["kernel_norm"].astype(float)
    C = results["contribution"].astype(float)
    theta = results["swc_map"].astype(float)

    # Diagnostic-only maps
    KT = K * theta
    KT[np.isnan(KT)] = np.nan
    KT_sum = np.nansum(KT)
    KT_share = (KT / KT_sum) if KT_sum > 0 else KT
    Delta = KT_share - K

    normK, epsK = robust_norm_for_contribution(K, mode=norm_mode)
    normC, epsC = robust_norm_for_contribution(C, mode=norm_mode)

    # Delta diverging norm
    dmask = np.isfinite(Delta)
    dmax = float(np.nanpercentile(np.abs(Delta[dmask]), 99.0)) if np.any(dmask) else 1e-6
    dmax = max(dmax, 1e-10)
    dnorm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)

    fig, axes = plt.subplots(1, 3, figsize=(22.5, 6.5))

    base_title = f"{results['date']}"
    if results.get("pressure_hpa") is not None:
        base_title += f"\nP={results['pressure_hpa']:.1f} hPa, sP={results['pressure_scale_sP']:.3f}"

    # ---------- (A) Kernel ----------
    ax = axes[0]
    im = ax.imshow(
        K + (epsK if epsK is not None else 0.0),
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="hot", norm=normK, interpolation="bilinear"
    )

    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, linewidth=0.8, alpha=0.18, color="white", zorder=2)

    # ✅ Kernel 86% mass contour (should be circular if kernel truly radial)
    draw_mass_contour(ax, Xi, Yi, K, level=mass_level, color="cyan", lw=boundary_lw, alpha=0.95, zorder=7)

    ax.plot(0, 0, "w+", markersize=18, markeredgewidth=3, zorder=8)
    _base_axes_setup(ax, max_extent, f"(A) Pure Kernel K\n{base_title}\nMass-contour: {mass_level:.2f}")
    ax.grid(True, alpha=0.25, linestyle="--", color="white")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Kernel weight (sum=1)", fontsize=12, fontweight="bold")

    # ---------- (B) Contribution ----------
    ax = axes[1]
    im = ax.imshow(
        C + (epsC if epsC is not None else 0.0),
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="hot", norm=normC, interpolation="bilinear"
    )

    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, linewidth=0.8, alpha=0.18, color="white", zorder=2)

    # ✅ Contribution 86% mass contour (this is your "amoeba", but now as a TRUE contour)
    draw_mass_contour(ax, Xi, Yi, C, level=mass_level, color="red", lw=boundary_lw, alpha=0.98, zorder=7)

    # (옵션) kernel contour도 같이 보여주면 비교가 매우 명확함
    draw_mass_contour(ax, Xi, Yi, K, level=mass_level, color="cyan", lw=boundary_lw, alpha=0.85, zorder=6)

    ax.plot(0, 0, "w+", markersize=18, markeredgewidth=3, zorder=8)
    _base_axes_setup(ax, max_extent, f"(B) Contribution C\n{base_title}\nRed=C, Cyan=K mass-contour @ {mass_level:.2f}")
    ax.grid(True, alpha=0.25, linestyle="--", color="white")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Contribution (sum=1)", fontsize=12, fontweight="bold")

    # ---------- (C) Delta ----------
    ax = axes[2]
    im = ax.imshow(
        Delta,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="RdBu_r", norm=dnorm, interpolation="bilinear"
    )

    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, linewidth=0.8, alpha=0.18, color="black", zorder=2)

    # ✅ Overlay: contribution contour (black) for reference
    draw_mass_contour(ax, Xi, Yi, C, level=mass_level, color="black", lw=boundary_lw, alpha=0.95, zorder=7)

    ax.plot(0, 0, "k+", markersize=18, markeredgewidth=3, zorder=8)
    _base_axes_setup(ax, max_extent, f"(C) Redistribution Δ = (K×θ share) - K\n{base_title}\nBlack=C mass-contour @ {mass_level:.2f}")
    ax.grid(True, alpha=0.25, linestyle="--", color="gray")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ share", fontsize=12, fontweight="bold")

    fig.suptitle("CRNP Footprint Diagnostics: TRUE 86% Mass Contours (Kernel vs Contribution)", fontsize=18, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel_depth_layers(results, save_path):
    dp = results['depth_profile']
    layers = dp['layer_contributions']

    fig, ax = plt.subplots(figsize=(8, 6))

    layer_names = [l['layer'] for l in reversed(layers)]
    contributions = [l['contribution_pct'] for l in reversed(layers)]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.barh(layer_names, contributions, color=colors[:len(layer_names)], alpha=0.7)

    for i, (bar, val) in enumerate(zip(bars, contributions)):
        ax.text(val + 1, i, f'{val:.1f}%',
                va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Contribution (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Depth Layer', fontsize=14, fontweight='bold')
    ax.set_title(f"Vertical Layer Contributions\n{results['date']}\nD₈₆ = {dp['D86']:.1f} cm",
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(contributions) * 1.15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Cross-section (unchanged from your version)
# ============================================================

def plot_panel_footprint_crosssection(results, save_path,
                                      bg_threshold=0.0005,
                                      gamma=0.45,
                                      n_levels=6):
    import matplotlib.patheffects as pe
    from matplotlib.colors import PowerNorm

    max_extent = results["max_extent"]
    dp = results['depth_profile']

    D86 = dp['D86']
    D_theo = dp['D']
    theta_used = dp['theta_used']
    rho_used = dp['bulk_density_used']

    # use kernel circle as representative horizontal scale (truth baseline)
    R86_mean = results.get("R86_radius_K", max_extent * 0.7)

    max_r_display = max(1.2 * R86_mean, 60.0)
    max_r_display = min(max_r_display, max_extent)

    max_d_display = max(2.0 * D86, 100.0)
    max_d_display = min(max_d_display, 100.0)

    distances = np.linspace(-max_r_display, max_r_display, 320)
    depths = np.linspace(0, max_d_display, 260)

    D_grid, R_grid = np.meshgrid(depths, distances)

    r_abs = np.abs(R_grid)
    Wr = (30 * np.exp(-r_abs / 1.6) + np.exp(-r_abs / 100.0)) * (1 - np.exp(-3.7 * r_abs))
    Wd = np.exp(-2.0 * D_grid / D_theo)

    W_combined = Wr * Wd
    W_combined = W_combined / (W_combined.max() + 1e-12)

    W_masked = np.ma.masked_less(W_combined, bg_threshold)

    fig, ax = plt.subplots(figsize=(20, 9), facecolor="white")
    ax.set_facecolor("white")

    cmap = plt.get_cmap("hot_r").copy()
    cmap.set_bad(color="white", alpha=1.0)

    levels = np.linspace(0.0, 1.0, n_levels)
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)

    im = ax.contourf(R_grid, D_grid, W_masked,
                     levels=levels, cmap=cmap, norm=norm, alpha=1.0)

    key_levels = [0.01, 0.05, 0.1, 0.5, 0.86]
    contours = ax.contour(R_grid, D_grid, W_masked,
                          levels=key_levels,
                          colors='black',
                          linewidths=1.2,
                          alpha=0.85)

    cl = ax.clabel(contours, levels=key_levels,
                   inline=True, fontsize=10, fmt='%.2f',
                   inline_spacing=10, use_clabeltext=True, colors='black')
    for t in cl:
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.axvline(R86_mean, color='black', linestyle='--', linewidth=2,
               label=f'R₈₆(K) ≈ {R86_mean:.0f}m', alpha=0.97, zorder=5)
    ax.axvline(-R86_mean, color='black', linestyle='--', linewidth=2, alpha=0.97, zorder=5)

    ax.axhline(D86, color='gold', linestyle='--', linewidth=4,
               label=f'D₈₆ = {D86:.1f}cm', alpha=0.97, zorder=5)

    ax.axhline(D_theo, color='orange', linestyle=':', linewidth=3.5,
               label=f'D = {D_theo:.1f}cm', alpha=0.9, zorder=5)

    ax.plot(0, 0, 'k+', markersize=16, markeredgewidth=3,
            label='CRNP Center', zorder=10)

    ax.set_xlabel('Distance from CRNP (m)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Depth below surface (cm)', fontsize=17, fontweight='bold')

    title_text = (f'CRNP Footprint Cross-Section: Wr(r) × Wd(d)\n'
                  f'Date: {results["date"]} | SWC θ={theta_used:.3f}, '
                  f'ρbd={rho_used:.2f} g/cm³\n'
                  f'Display: ±{max_r_display:.0f}m × 0-{max_d_display:.0f}cm '
                  f'(bg={bg_threshold:g}, gamma={gamma:g})')
    ax.set_title(title_text, fontsize=15, fontweight='bold', pad=20)

    ax.invert_yaxis()
    ax.set_xlim(-max_r_display, max_r_display)
    ax.set_ylim(max_d_display, 0)

    ax.grid(True, alpha=0.25, linestyle='--', color='gray', linewidth=0.8, zorder=2)
    ax.tick_params(labelsize=12)

    legend = ax.legend(loc='lower right', fontsize=13, framealpha=0.98,
                       edgecolor='black', fancybox=True, shadow=True,
                       borderpad=1)
    legend.set_zorder(20)

    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02, aspect=40)
    cbar.set_label('Normalized Combined Sensitivity\nWr(r) × Wd(d)',
                   fontsize=13, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_all_panels(analyzer, results, out_dir, prefix="crnp"):
    _ensure_dir(out_dir)
    d = results["date"].replace("-", "")

    print(f"\nSaving panels for {results['date']}...")

    plot_panel_swc(analyzer, results,
                   os.path.join(out_dir, f"{prefix}_{d}_01_swc_distribution.png"))

    plot_panel_veg(results,
                   os.path.join(out_dir, f"{prefix}_{d}_02_vegetation_map.png"))

    plot_panel_footprint_diagnostics_3up(
        results,
        os.path.join(out_dir, f"{prefix}_{d}_03_kernel_kTheta_delta.png"),
        norm_mode="log",
        boundary_lw=2.0
    )

    plot_panel_depth_layers(results,
                            os.path.join(out_dir, f"{prefix}_{d}_04_depth_layers.png"))

    plot_panel_footprint_crosssection(
        results,
        os.path.join(out_dir, f"{prefix}_{d}_05_crosssection.png"),
        bg_threshold=0.0005,
        gamma=0.45
    )

    print(f"✓ Saved 5 panels to: {out_dir}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CRNP FOOTPRINT ANALYSIS (HORIZONTAL + VERTICAL)")
    print("=" * 70)

    geo_path = r"E:\02.Data\05.CRNP\02.UTS_system\98.Code_test\99.Data\01.HC\01.Input\geo_locations.xlsx"
    swc_path = r"E:\02.Data\05.CRNP\02.UTS_system\98.Code_test\99.Data\01.HC\02.Output\01.Preprocessed\PC_FDR_daily_depths.xlsx"
    pressure_path = r"E:\02.Data\05.CRNP\02.UTS_system\98.Code_test\99.Data\01.HC\02.Output\01.Preprocessed\HC_CRNP_input.xlsx"
    out_dir = r"E:\02.Data\05.CRNP\02.UTS_system\98.Code_test\01.FootprintRadius\Footprint_Depth\HC"

    analyzer = CRNPAnalyzer(geo_path, swc_path, pressure_path=pressure_path)

    test_dates = [
        "2024-08-16",
        "2024-09-20",
        "2024-10-20",
        "2025-10-26",
    ]

    use_pressure = True
    P0 = 1013.25
    alpha_p = 1.0

    edge_control = True
    outside_fill = "nearest"
    hull_buffer_m = 150.0

    bulk_density = 1.22

    clip_min = 0.05
    clip_max = 0.50

    # ✅ 균일이면 '완전 원'만 보이게 하는 기준 (필요하면 튜닝)
    circle_l1_tol = 3e-3

    print(f"\nAnalysis Settings:")
    print(f"- Bulk density: {bulk_density} g/cm³")
    print(f"- Pressure correction: {use_pressure}")
    print(f"- Edge control: {edge_control} ({outside_fill}, buffer={hull_buffer_m}m)")
    print(f"- Interp clip: {clip_min}-{clip_max}")
    print(f"- Circle L1 tol: {circle_l1_tol:.2e}")

    for date_str in test_dates:
        print(f"\n{'='*70}")
        print(f"Analyzing {date_str}...")
        print(f"{'='*70}")

        try:
            results = analyzer.analyze_single_day(
                date_str,
                base_veg_height=0.3,
                max_extent=150,
                resolution=5,
                ddeg=15,
                use_pressure=use_pressure,
                P0=P0,
                alpha_p=alpha_p,
                edge_control=edge_control,
                hull_buffer_m=hull_buffer_m,
                outside_fill=outside_fill,
                bulk_density=bulk_density,
                clip_min=clip_min,
                clip_max=clip_max,
                fill_remaining_nans_with_nearest=True,
                circle_l1_tol=circle_l1_tol
            )

            print_footprint_diagnostics(results)

            print(f"\n🌍 HORIZONTAL FOOTPRINT:")
            print(f"  ✓ Predicted SWC: {results['predicted_swc']:.4f} cm³/cm³")
            print(f"  ✓ Mean SWC: {results['mean_swc']:.4f} cm³/cm³")
            print(f"  ✓ Valid sensors: {results['n_valid_sensors']}/{len(analyzer.sensor_ids)}")
            print(f"  ✓ R86(Kernel circle): {results['R86_radius_K']:.1f} m")
            print(f"  ✓ R86(C radial): {results['R86_radius_C']:.1f} m")
            print(f"  ✓ Uniform-effective?: {results['uniform_effective']}")

            dp = results['depth_profile']
            print(f"\n📏 VERTICAL FOOTPRINT (at r={dp['distance_r']:.0f}m):")
            print(f"  ✓ θ used: {dp['theta_used']:.4f} (footprint-weighted average)")
            print(f"  ✓ ρbd used: {dp['bulk_density_used']:.2f} g/cm³")
            print(f"  ✓ Penetration depth D: {dp['D']:.1f} cm")
            print(f"  ✓ D86 (86% depth): {dp['D86']:.1f} cm")

            if results.get("pressure_hpa") is not None:
                print(f"\n🌡️  PRESSURE:")
                print(f"  ✓ {results['pressure_hpa']:.2f} hPa (scale factor sP={results['pressure_scale_sP']:.4f})")

            save_all_panels(analyzer, results, out_dir, prefix="crnp")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("Analysis completed!")
    print(f"Output directory: {out_dir}")
    print(f"Total panels per date: 5")
    print(f"{'='*70}")
