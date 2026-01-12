"""
Panel 4: 깊이 레이어 기여도 (막대 그래프)
"""
import matplotlib.pyplot as plt

def plot_panel_depth_layers(results, save_path, config):
    """
    Panel 4: Layer contributions bar chart
    
    Args:
        results: analyze_single_day 결과
        save_path: 저장 경로
        config: Config 객체
    """
    dp = results['depth_profile']
    layers = dp['layer_contributions']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 레이어 이름과 기여도 (역순 - 위에서 아래로)
    layer_names = [l['layer'] for l in reversed(layers)]
    contributions = [l['contribution_pct'] for l in reversed(layers)]
    
    # Config에서 색상 가져오기
    colors = config.plotting['panel_depth']['colors']
    
    # 막대 그래프
    bars = ax.barh(layer_names, contributions, 
                   color=colors[:len(layer_names)], alpha=0.7)
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, contributions)):
        ax.text(val + 1, i, f'{val:.1f}%',
                va='center', fontsize=12, fontweight='bold')
    
    # 축 설정
    ax.set_xlabel('Contribution (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Depth Layer', fontsize=14, fontweight='bold')
    ax.set_title(
        f"Vertical Layer Contributions\n{results['date']}\nD₈₆ = {dp['D86']:.1f} cm",
        fontsize=16, fontweight='bold'
    )
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(contributions) * 1.15)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)