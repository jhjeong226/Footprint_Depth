"""
Panel 5: 2D Cross-section (Distance × Depth) - Cumulative Contribution View
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm

def plot_panel_footprint_crosssection(results, save_path, config,
                                      bg_threshold=0.001):
    """
    Panel 5: 2D Cross-section with Cumulative Contribution

    개선 사항:
    - 누적 기여도(%) 기반 컨투어: 50%, 70%, 86%, 95%
    - 컨투어 레벨에 따른 색상 구분
    - D86, R86 정보를 직관적으로 표시

    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
        bg_threshold: 배경 임계값
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

    max_d_display = max(2.0 * D86, 50.0)
    max_d_display = min(max_d_display, 80.0)

    # 그리드 생성
    distances = np.linspace(-max_r_display, max_r_display, 400)
    depths = np.linspace(0, max_d_display, 300)

    D_grid, R_grid = np.meshgrid(depths, distances)

    # Wr (수평) - radial intensity function
    r_abs = np.abs(R_grid)
    Wr = (30 * np.exp(-r_abs / 1.6) + np.exp(-r_abs / 100.0)) * (1 - np.exp(-3.7 * r_abs))

    # Wd (수직)
    Wd = np.exp(-2.0 * D_grid / D_theo)

    # 결합 감도
    W_combined = Wr * Wd

    # 누적 기여도 계산 (벡터화된 방식)
    W_flat = W_combined.flatten()
    total_sum = W_flat.sum()

    # 정렬 기반 누적 기여도 계산 (빠른 방식)
    sorted_indices = np.argsort(W_flat)[::-1]  # 큰 값부터
    sorted_weights = W_flat[sorted_indices]
    cumsum = np.cumsum(sorted_weights) / total_sum

    # 원래 위치로 복원
    cumulative_flat = np.zeros_like(W_flat)
    cumulative_flat[sorted_indices] = cumsum
    cumulative = cumulative_flat.reshape(W_combined.shape)

    # Figure 생성
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="white")
    ax.set_facecolor("white")

    # 누적 기여도 레벨 (%)
    contribution_levels = [0.50, 0.70, 0.86, 0.95, 1.0]
    level_labels = ['50%', '70%', '86%', '95%', '100%']

    # 색상 정의 (진한색 → 연한색)
    colors = ['#8B0000', '#FF4500', '#FFA500', '#FFD700', '#FFFACD']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    norm = BoundaryNorm(contribution_levels, cmap.N)

    # 배경 마스킹
    cumulative_masked = np.ma.masked_where(cumulative > 0.99, cumulative)

    # Contourf로 영역 채우기
    im = ax.contourf(R_grid, D_grid, cumulative_masked,
                     levels=[0] + contribution_levels,
                     colors=['#8B0000', '#FF4500', '#FFA500', '#FFD700', '#FFFACD'],
                     alpha=0.85)

    # 컨투어 라인 (각 레벨별로 색상 지정)
    contour_colors = ['darkred', 'orangered', 'darkorange', 'gold']
    for level, color, label in zip(contribution_levels[:-1], contour_colors, level_labels[:-1]):
        cs = ax.contour(R_grid, D_grid, cumulative,
                       levels=[level],
                       colors=[color],
                       linewidths=2.5,
                       linestyles='solid')
        # 라벨 추가 (딕셔너리 포맷터 사용)
        fmt_dict = {level: label}
        ax.clabel(cs, inline=True, fontsize=11, fmt=fmt_dict, inline_spacing=15)

    # ========== 측정 깊이 정보 강조 ==========

    # D86 라인 (가장 중요한 정보)
    ax.axhline(D86, color='blue', linestyle='-', linewidth=3,
               label=f'D₈₆ = {D86:.1f} cm (86% signal depth)', alpha=0.9, zorder=10)

    # D (1/e² depth) 라인
    ax.axhline(D_theo, color='navy', linestyle='--', linewidth=2,
               label=f'D = {D_theo:.1f} cm (penetration depth)', alpha=0.8, zorder=9)

    # R86 라인
    ax.axvline(R86_mean, color='green', linestyle='-', linewidth=2.5,
               label=f'R₈₆ = {R86_mean:.0f} m (86% radius)', alpha=0.9, zorder=10)
    ax.axvline(-R86_mean, color='green', linestyle='-', linewidth=2.5,
               alpha=0.9, zorder=10)

    # ========== 레이어별 기여도 정보 ==========
    layer_info = dp.get('layer_contributions', [])

    # 레이어 경계와 기여도 표시
    layer_boundaries = [0, 10, 20, 40]
    y_offset = max_r_display * 0.02

    for i, layer in enumerate(layer_info[:3]):  # 상위 3개 레이어만
        depth_start = layer_boundaries[i] if i < len(layer_boundaries) else 0
        depth_end = layer_boundaries[i+1] if i+1 < len(layer_boundaries) else 100
        contrib = layer['contribution_pct']

        # 레이어 영역 표시
        ax.axhline(depth_end, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # 기여도 텍스트 (오른쪽에 표시)
        mid_depth = (depth_start + depth_end) / 2
        if mid_depth < max_d_display:
            ax.text(max_r_display * 0.98, mid_depth,
                   f'{layer["layer"]}: {contrib:.1f}%',
                   fontsize=10, fontweight='bold',
                   va='center', ha='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.9, edgecolor='gray'),
                   zorder=15)

    # CRNP center
    ax.plot(0, 0, 'k+', markersize=12, markeredgewidth=2.5,
            label='CRNP Center', zorder=15)

    # 축 설정
    ax.set_xlabel('Distance from CRNP (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Depth (cm)', fontsize=13, fontweight='bold')

    # 제목
    title_text = f'CRNP Measurement Volume | {results["date"]}'
    subtitle = f'θ = {theta_used:.2f} cm³/cm³, ρ = {rho_used:.2f} g/cm³'
    ax.set_title(f'{title_text}\n{subtitle}', fontsize=14, fontweight='bold', pad=10)

    ax.invert_yaxis()
    ax.set_xlim(-max_r_display, max_r_display)
    ax.set_ylim(max_d_display, 0)

    ax.grid(True, alpha=0.3, linestyle='--', color='gray', linewidth=0.5, zorder=1)
    ax.tick_params(labelsize=11)

    # Legend (위치 조정)
    legend = ax.legend(loc='lower left', fontsize=11, framealpha=0.95,
                       edgecolor='gray', fancybox=True,
                       borderpad=0.8)
    legend.set_zorder(20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, aspect=30)
    cbar.set_label('Cumulative Signal\nContribution', fontsize=12, fontweight='bold')
    cbar.set_ticks([0.25, 0.60, 0.78, 0.905, 0.975])
    cbar.set_ticklabels(['0-50%', '50-70%', '70-86%', '86-95%', '95-100%'])
    cbar.ax.tick_params(labelsize=10)

    # 해석 안내 텍스트
    interpretation = (
        "Color zones show cumulative signal contribution:\n"
        "• Dark red (0-50%): Core measurement zone\n"
        "• Orange (50-86%): Primary measurement zone\n"
        "• Yellow (86-95%): Secondary contribution"
    )
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
