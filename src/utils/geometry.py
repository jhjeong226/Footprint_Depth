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