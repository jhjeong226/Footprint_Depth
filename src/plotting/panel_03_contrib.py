"""
Panel 3: Footprint diagnostics (3-up layout with TRUE mass contours)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from .base import setup_base_axes, robust_norm_for_contribution
from ..utils.geometry import plot_sector_rays, mass_contour_field, draw_mass_contour

def plot_panel_footprint_diagnostics_3up(results, save_path, config,
                                        show_sector_rays=True,
                                        norm_mode="log",
                                        boundary_lw=2.5,
                                        mass_level=0.86):
    """
    Panel 3 (3-up) with TRUE mass-contours:
    
    (A) Pure kernel K + its 86% mass contour (원형, 토양수분 영향 없음)
    (B) Contribution C + its 86% mass contour (아메바, 토양수분 재분배 효과)
    (C) Delta (K×θ share - K) + C contour overlay (재분배 크기 시각화)
    
    이것이 실제 Actual Signal Contribution을 표현합니다!
    """
    max_extent = results["max_extent"]
    ddeg = results["ddeg"]
    Xi, Yi = results["Xi"], results["Yi"]

    K = results["kernel_norm"].astype(float)
    C = results["contribution"].astype(float)
    theta = results["swc_map"].astype(float)

    # Diagnostic maps
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

    # ========== (A) Pure Kernel K ==========
    ax = axes[0]
    im = ax.imshow(
        K + (epsK if epsK is not None else 0.0),
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="hot", norm=normK, interpolation="bilinear"
    )

    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, linewidth=0.8, alpha=0.18, color="white", zorder=2)

    # ✅ Kernel 86% mass contour (항상 거의 원형)
    draw_mass_contour(ax, Xi, Yi, K, level=mass_level, color="cyan", lw=boundary_lw, alpha=0.95, zorder=7)

    ax.plot(0, 0, "w+", markersize=18, markeredgewidth=3, zorder=8)
    
    title_A = f"(A) Pure Kernel K\n{base_title}\n(No soil moisture effect)"
    ax.set_title(title_A, fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", color="white")
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Kernel weight (sum=1)", fontsize=11, fontweight="bold")

    # ========== (B) Actual Signal Contribution C ==========
    ax = axes[1]
    im = ax.imshow(
        C + (epsC if epsC is not None else 0.0),
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="hot", norm=normC, interpolation="bilinear"
    )

    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, linewidth=0.8, alpha=0.18, color="white", zorder=2)

    # ✅ Contribution 86% mass contour (토양수분 재분배 효과 반영, "아메바")
    draw_mass_contour(ax, Xi, Yi, C, level=mass_level, color="red", lw=boundary_lw, alpha=0.98, zorder=7)

    # ✅ Kernel contour도 같이 표시 (비교용)
    draw_mass_contour(ax, Xi, Yi, K, level=mass_level, color="cyan", lw=boundary_lw*0.8, alpha=0.7, zorder=6)

    ax.plot(0, 0, "w+", markersize=18, markeredgewidth=3, zorder=8)
    
    title_B = f"(B) Actual Signal Contribution C\n{base_title}\n(Red=C, Cyan=K mass-contour @ {mass_level:.0%})"
    ax.set_title(title_B, fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", color="white")
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Contribution (sum=1)", fontsize=11, fontweight="bold")

    # ========== (C) Delta (Redistribution) ==========
    ax = axes[2]
    im = ax.imshow(
        Delta,
        extent=[-max_extent, max_extent, -max_extent, max_extent],
        origin="lower", cmap="RdBu_r", norm=dnorm, interpolation="bilinear"
    )

    if show_sector_rays:
        plot_sector_rays(ax, max_extent, ddeg, linewidth=0.8, alpha=0.18, color="black", zorder=2)

    # ✅ Contribution contour overlay (검은색, 참고용)
    draw_mass_contour(ax, Xi, Yi, C, level=mass_level, color="black", lw=boundary_lw, alpha=0.95, zorder=7)

    ax.plot(0, 0, "k+", markersize=18, markeredgewidth=3, zorder=8)
    
    title_C = f"(C) Redistribution Δ = (K×θ share) - K\n{base_title}\n(Black=C mass-contour @ {mass_level:.0%})"
    ax.set_title(title_C, fontsize=14, fontweight="bold")
    ax.set_xlabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", color="gray")
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ share", fontsize=11, fontweight="bold")

    # 전체 제목
    fig.suptitle(
        "CRNP Footprint: Kernel vs Actual Signal Contribution (TRUE 86% Mass Contours)",
        fontsize=16, fontweight="bold", y=0.98
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Backward compatibility: 기존 함수명 유지
plot_panel_footprint_amoeba = plot_panel_footprint_diagnostics_3up