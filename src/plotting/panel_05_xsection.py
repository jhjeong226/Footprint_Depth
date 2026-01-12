"""
Panel 5: 2D Cross-section (Distance × Depth combined sensitivity)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm

def plot_panel_footprint_crosssection(results, save_path, config,
                                      bg_threshold=0.0005,
                                      gamma=0.45,
                                      n_levels=6):
    """
    Panel 5: 2D Cross-section (Distance × Depth combined sensitivity)
    
    개선 사항:
    - bg_threshold 이하 값을 0으로 클리핑 → 0(=배경)이 흰색
    - contour line: 검정색
    - contour label: 검정 글씨 + 흰색 외곽선 → 어디서든 잘 보임
    
    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
        bg_threshold: 배경 임계값
        gamma: PowerNorm gamma 값
        n_levels: contour 레벨 수
    """
    max_extent = results["max_extent"]
    dp = results['depth_profile']
    
    D86 = dp['D86']
    D_theo = dp['D']
    theta_used = dp['theta_used']
    rho_used = dp['bulk_density_used']
    
    # R86 평균
    rps = [s["rp"] for s in results["R86_phi_sectors"]]
    R86_mean = np.mean(rps) if len(rps) > 0 else max_extent * 0.7
    
    # 표시 범위
    max_r_display = max(1.2 * R86_mean, 60.0)
    max_r_display = min(max_r_display, max_extent)
    
    max_d_display = max(2.0 * D86, 100.0)
    max_d_display = min(max_d_display, 100.0)
    
    # 그리드 생성
    distances = np.linspace(-max_r_display, max_r_display, 320)
    depths = np.linspace(0, max_d_display, 260)
    
    D_grid, R_grid = np.meshgrid(depths, distances)
    
    # Wr (수평) - 원본 코드의 radial intensity function
    r_abs = np.abs(R_grid)
    Wr = (30 * np.exp(-r_abs / 1.6) + np.exp(-r_abs / 100.0)) * (1 - np.exp(-3.7 * r_abs))
    
    # Wd (수직)
    Wd = np.exp(-2.0 * D_grid / D_theo)
    
    # 결합 감도
    W_combined = Wr * Wd
    W_combined = W_combined / (W_combined.max() + 1e-12)
    
    # 배경 흰색 처리
    W_masked = np.ma.masked_less(W_combined, bg_threshold)
    W_plot = W_combined.copy()
    W_plot[W_plot < bg_threshold] = 0.0
    
    # Figure 생성
    fig, ax = plt.subplots(figsize=(20, 9), facecolor="white")
    ax.set_facecolor("white")
    
    # Colormap
    cmap = plt.get_cmap("hot_r").copy()  # 0 -> white
    cmap.set_bad(color="white", alpha=1.0)
    
    # Contourf
    levels = np.linspace(0.0, 1.0, n_levels)
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)
    
    im = ax.contourf(R_grid, D_grid, W_masked,
                     levels=levels, cmap=cmap, norm=norm, alpha=1.0)
    
    # Contour lines (검정색)
    key_levels = [0.001, 0.01, 0.05, 0.1, 0.5, 0.86]
    contours = ax.contour(R_grid, D_grid, W_masked,
                          levels=key_levels,
                          colors='black',
                          linewidths=[0.9, 1.0, 1.1, 1.25, 1.4, 1.6],
                          alpha=0.85)
    
    # Contour labels (검정 글씨 + 흰 외곽선)
    cl = ax.clabel(contours, levels=[0.001, 0.01, 0.05, 0.1, 0.5, 0.86],
                   inline=True, fontsize=10, fmt='%.3f',
                   inline_spacing=10, use_clabeltext=True, colors='black')
    for t in cl:
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
    
    # R86, D86, D 표시
    ax.axvline(R86_mean, color='black', linestyle='--', linewidth=2,
               label=f'R₈₆ ≈ {R86_mean:.0f}m (86% radius)', alpha=0.97, zorder=5)
    ax.axvline(-R86_mean, color='black', linestyle='--', linewidth=2, 
               alpha=0.97, zorder=5)
    
    ax.axhline(D86, color='gold', linestyle='--', linewidth=4,
               label=f'D₈₆ = {D86:.1f}cm (86% depth)', alpha=0.97, zorder=5)
    
    ax.axhline(D_theo, color='orange', linestyle=':', linewidth=3.5,
               label=f'D = {D_theo:.1f}cm (1/e² depth)', alpha=0.9, zorder=5)
    
    # Layer markers
    for layer_depth, label in [(5, '5cm'), (10, '10cm'), (15, '15cm'),
                               (20, '20cm'), (25, '25cm'), (30, '30cm'),
                               (40, '40cm')]:
        if layer_depth < max_d_display:
            ax.axhline(layer_depth, color='gray', linestyle=':',
                       linewidth=1.0, alpha=0.45, zorder=1)
            ax.text(max_r_display * 0.99, layer_depth, f' {label}',
                    fontsize=10, color='dimgray', va='center', ha='right',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              alpha=0.9, edgecolor='none'))
    
    # CRNP center
    ax.plot(0, 0, 'k+', markersize=16, markeredgewidth=3,
            label='CRNP Center', zorder=10)
    
    # 축 설정
    ax.set_xlabel('Distance from CRNP (m)', fontsize=17, fontweight='bold')
    ax.set_ylabel('Depth below surface (cm)', fontsize=17, fontweight='bold')
    
    title_text = (f'CRNP Footprint Cross-Section: Combined Horizontal × Vertical Sensitivity\n'
                  f'Date: {results["date"]} | SWC θ={theta_used:.3f} cm³/cm³, '
                  f'Bulk Density ρ={rho_used:.2f} g/cm³\n'
                  f'Display Range: ±{max_r_display:.0f}m × 0-{max_d_display:.0f}cm '
                  f'(bg threshold={bg_threshold:g}, gamma={gamma:g})')
    ax.set_title(title_text, fontsize=15, fontweight='bold', pad=20)
    
    ax.invert_yaxis()
    ax.set_xlim(-max_r_display, max_r_display)
    ax.set_ylim(max_d_display, 0)
    
    ax.grid(True, alpha=0.25, linestyle='--', color='gray', linewidth=0.8, zorder=2)
    ax.tick_params(labelsize=12)
    
    # Legend
    legend = ax.legend(loc='lower right', fontsize=13, framealpha=0.98,
                       edgecolor='black', fancybox=True, shadow=True,
                       borderpad=1)
    legend.set_zorder(20)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.038, pad=0.02, aspect=40)
    cbar.set_label('Normalized Combined Sensitivity\nWr(r) × Wd(d)',
                   fontsize=13, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)