"""
기하학적 계산 유틸리티
"""
import numpy as np
from scipy.spatial import Delaunay

def convex_hull_mask(xy_points, Xi, Yi, buffer_m=0.0):
    """
    Convex hull 내부만 True 반환
    
    Args:
        xy_points: 센서 좌표 (N, 2)
        Xi, Yi: 그리드 좌표
        buffer_m: Hull 확장 버퍼 (m)
    
    Returns:
        mask: Xi, Yi와 같은 shape의 boolean 배열
    """
    pts = np.asarray(xy_points, float)
    if pts.shape[0] < 3:
        return np.ones_like(Xi, dtype=bool)
    
    # 버퍼 적용
    if buffer_m > 0:
        r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        scale = (r + buffer_m) / (r + 1e-12)
        pts = pts * scale[:, None]
    
    # Delaunay 삼각분할
    tri = Delaunay(pts)
    test = np.c_[Xi.ravel(), Yi.ravel()]
    inside = tri.find_simplex(test) >= 0
    
    return inside.reshape(Xi.shape)


def build_amoeba_boundary_polyline(sectors, arc_step_deg=1.0):
    """
    30° 섹터별 R86(φ)로 연속 경계선 생성

    30° 섹터별 rp로 연속 경계선 생성 (polyline)
    
    NOTE: 이것은 directional R86 값들을 연결한 것으로,
    TRUE mass contour와는 다릅니다!
    실제 신호 기여 경계를 보려면 mass_contour_field와 
    draw_mass_contour를 사용하세요.
    
    Args:
        sectors: [{'phi0': 0, 'phi1': 30, 'rp': 120}, ...]
        arc_step_deg: 호 간격 (도)
    
    Returns:
        xs, ys: 경계선 좌표
    """
    if not sectors:
        return np.array([]), np.array([])
    
    sectors = sorted(sectors, key=lambda s: s["phi0"])
    xs, ys = [], []
    n = len(sectors)
    
    for i in range(n):
        s = sectors[i]
        phi0, phi1, rp = s["phi0"], s["phi1"], s["rp"]
        
        # 호 그리기
        phis = np.arange(phi0, phi1 + arc_step_deg, arc_step_deg)
        phir = np.deg2rad(phis)
        xs.extend((rp * np.cos(phir)).tolist())
        ys.extend((rp * np.sin(phir)).tolist())
        
        # 다음 섹터로 연결
        s_next = sectors[(i + 1) % n]
        rp_next = s_next["rp"]
        phib = np.deg2rad(phi1 % 360.0)
        
        xs.append(rp * np.cos(phib))
        ys.append(rp * np.sin(phib))
        xs.append(rp_next * np.cos(phib))
        ys.append(rp_next * np.sin(phib))
    
    # 닫기
    xs.append(xs[0])
    ys.append(ys[0])
    
    return np.array(xs), np.array(ys)


def plot_sector_rays(ax, max_extent, ddeg, **plot_kwargs):
    """
    섹터 경계 레이 그리기
    
    Args:
        ax: matplotlib axis
        max_extent: 최대 반경
        ddeg: 섹터 각도
        plot_kwargs: 플롯 옵션
    """
    for ang in np.arange(0, 360, ddeg):
        rad = np.deg2rad(ang)
        ax.plot(
            [0, max_extent * np.cos(rad)],
            [0, max_extent * np.sin(rad)],
            **plot_kwargs
        )

def mass_contour_field(W, eps=1e-30):
    """
    W(=kernel or contribution)가 주어졌을 때,
    각 픽셀 i에 대해:
      Q_i = sum(W_j where W_j >= W_i)
    를 계산한 필드 Q를 반환.
    
    즉, Q=0.86 등고선은 '상위 기여 영역이 86%가 되는 경계' (HDR mass contour).
    이것이 TRUE contour입니다!
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
    
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    Q_flat = csum[inv]
    return Q_flat.reshape(W.shape)


def draw_mass_contour(ax, Xi, Yi, W, level=0.86, color="red", lw=2.5, alpha=1.0, zorder=7):
    """
    W로부터 HDR mass contour (Q=level)을 계산해서 contour line을 그림.
    이것이 실제 86% 신호 기여 경계선입니다!
    """
    Q = mass_contour_field(W)
    cs = ax.contour(Xi, Yi, Q, levels=[level], colors=[color], linewidths=lw, alpha=alpha, zorder=zorder)
    return cs