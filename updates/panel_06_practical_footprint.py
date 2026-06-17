"""
Panel 6: Practical Footprint (Schrön 2023) — R86 vs 실용적 감지 거리
=====================================================================

두 개의 서브플롯:
  (A) 기여도 맵 위에 세 개의 반경을 겹쳐 그림:
        - R86 (해석적, 점선)         : 기존 footprint 정의
        - R_practical (wetting +10%) : Δθ>0 감지 최대 거리
        - R_practical (drying -10%)  : Δθ<0 감지 최대 거리
  (B) 논문 Fig.8b 스타일 sensitivity 곡선:
        주 밭 토양수분 θ1 vs 감지 거리 R (Δθ = 5/10/20%),
        현재 관측 θ를 수직선으로 표시.

기존 패널과 동일한 호출 규약: plot_panel_practical_footprint(results, save_path, config, sigcon)
sigcon: SignalContribution 인스턴스 (analyzer.sigcon)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap


def _draw_circle(ax, radius, color, label, ls="--", lw=2.5):
    th = np.linspace(0, 2 * np.pi, 360)
    ax.plot(radius * np.cos(th), radius * np.sin(th),
            color=color, ls=ls, lw=lw, label=label, zorder=6,
            path_effects=[pe.withStroke(linewidth=lw + 1.5, foreground="white")])


def plot_panel_practical_footprint(results, save_path, config, sigcon,
                                   h=5.0, dtheta=0.10, sigma_N=0.01,
                                   cmap_name="Reds"):
    """
    Schrön 2023 실용적 footprint 시각화.

    Args:
        results : analyze_single_day 결과
        save_path : 저장 경로
        config : Config
        sigcon : SignalContribution 인스턴스
        h : 절대습도 (g/m³)
        dtheta : 토양수분 변화량 (기본 ±0.10)
        sigma_N : 센서 정밀도 (기본 0.01 = 1%)
    """
    max_extent = results["max_extent"]
    Xi, Yi = results["Xi"], results["Yi"]
    C = results["contribution"].astype(float)
    theta = float(results["predicted_swc"])
    press = results.get("pressure_hpa") or 1013.25
    Hveg = float(results.get("veg_height", 0.0))

    # ── 거리 계산 ──────────────────────────────────────────────
    R86 = results.get("R86_analytical")
    if R86 is None or not np.isfinite(R86):
        R86 = sigcon.physics.calculate_analytical_R86(
            theta=theta, h=h, pressure_hpa=press, Hveg=Hveg)
    R_wet = sigcon.practical_footprint_distance(
        theta, dtheta=+abs(dtheta), sigma_N=sigma_N, h=h,
        pressure_hpa=press, Hveg=Hveg)
    R_dry = sigcon.practical_footprint_distance(
        theta, dtheta=-abs(dtheta), sigma_N=sigma_N, h=h,
        pressure_hpa=press, Hveg=Hveg)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ════════════════ (A) 기여도 맵 + 반경 오버레이 ════════════════
    ax = axes[0]
    C_masked = np.ma.masked_where(~np.isfinite(C) | (C <= 0), C)
    if C_masked.count() > 0:
        vmax = float(np.nanpercentile(C_masked.compressed(), 99.0))
        levels = np.linspace(0, vmax, 12)
        ax.contourf(Xi, Yi, C_masked, levels=levels, cmap=cmap_name,
                    extend="max", alpha=0.9, zorder=1)

    _draw_circle(ax, R86, "#1f3a8a", f"R₈₆ (conventional) = {R86:.0f} m",
                 ls="--", lw=2.5)
    _draw_circle(ax, R_dry, "#b22222",
                 f"R practical, dry −{abs(dtheta)*100:.0f}% = {R_dry:.0f} m",
                 ls="-", lw=2.5)
    _draw_circle(ax, R_wet, "#1a9850",
                 f"R practical, wet +{abs(dtheta)*100:.0f}% = {R_wet:.0f} m",
                 ls="-", lw=2.5)

    ax.plot(0, 0, "k+", markersize=18, markeredgewidth=3, zorder=8)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distance (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"(A) R₈₆ vs Practical Footprint\n{results['date']}  "
        f"θ={theta*100:.1f}%, h={h:.0f} g/m³, σN={sigma_N*100:.0f}%",
        fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.25, ls="--")

    # ════════════════ (B) sensitivity 곡선 (Fig.8b 스타일) ════════════════
    ax = axes[1]
    theta1_axis = np.linspace(0.01, 0.50, 60)
    colors = {0.05: "#9ecae1", 0.10: "#4292c6", 0.20: "#08519c"}
    for dth, col in colors.items():
        R_curve = [
            sigcon.practical_footprint_distance(
                t1, dtheta=dth, sigma_N=sigma_N, h=h,
                pressure_hpa=press, Hveg=Hveg)
            for t1 in theta1_axis
        ]
        ax.plot(R_curve, theta1_axis * 100, color=col, lw=2.5,
                label=f"Δθ = +{dth*100:.0f}%")

    # R86 곡선(비교용, 점선)
    R86_curve = [
        sigcon.physics.calculate_analytical_R86(
            theta=t1, h=h, pressure_hpa=press, Hveg=Hveg)
        for t1 in theta1_axis
    ]
    ax.plot(R86_curve, theta1_axis * 100, color="#1f3a8a", lw=2.0,
            ls="--", label="R₈₆ (conventional)")

    # 현재 관측 θ 수평선
    ax.axhline(theta * 100, color="black", ls=":", lw=1.8, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.98, theta * 100 + 1,
            f"current θ = {theta*100:.1f}%", ha="right", fontsize=10,
            fontstyle="italic")

    ax.set_xlabel("Distance R to remote field (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Main field soil moisture θ₁ (vol %)", fontsize=12, fontweight="bold")
    ax.set_title("(B) Sensitivity to Remote Soil Moisture Changes\n"
                 "(detectable if ΔN > σN)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(260, np.nanmax(R86_curve) * 1.05))
    ax.set_ylim(0, 50)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle("CRNP Practical Footprint (Schrön et al. 2023)",
                 fontsize=16, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"R86": float(R86), "R_wet": float(R_wet), "R_dry": float(R_dry)}
