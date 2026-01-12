"""
플롯 공통 유틸리티
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

def setup_base_axes(ax, max_extent, title, xlabel="Distance (m)", ylabel="Distance (m)"):
    """기본 축 설정"""
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