import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import os
import sys
import argparse
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
# Paths
data_path = 'data/plain/current/2000/'
figs_path = 'figs/'
os.makedirs(figs_path, exist_ok=True)
 
# Foundation colour palette
FOUNDATION_COLORS = {
    'Care':      '#4CAF50',
    'Fairness':  '#2196F3',
    'Loyalty':   '#FF9800',
    'Authority': '#9C27B0',
    'Sanctity':  '#F44336',
    'Liberty':   '#795548',
}
 
MORAL_POS_COLOR = '#2ecc71'
MORAL_NEG_COLOR = '#e74c3c'
SHARED_COLOR    = '#B0BEC5'
 
TOP_N          = 15
LOADING_THRESH = 0.15
SHARED_THRESH  = 0.05
 
 
# ==============================================================================
# Data loading
# ==============================================================================
 
def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))
 
 
def find_char(char_df, name):
    """Return (col_index_into_RAW_A, full_name).
    char_df rows correspond to columns of RAW_A (characters are columns).
    """
    matches = char_df[char_df['character'].str.contains(name, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Character '{name}' not found.")
    return matches.index[0], matches.iloc[0]['character']
 
 
# ==============================================================================
# Core math
# ==============================================================================
 
def build_foundation_vector(U, Sigma, trait_map, num_comp):
    """
    Build a unit vector in k-space for this moral foundation.
 
    RAW_A = U @ diag(Sigma) @ V.T
    U : [n_traits x n_factors] -- each ROW is one trait in factor space
 
    We sum signed trait rows from U (scaled by Sigma) to build an ideal
    trait profile vector in factor space.
 
    trait_map : {trait_index (1-based): direction (+1/-1)}
    Returns ideal_v_k [num_comp], US [n_traits x num_comp]
    """
    US = U[:, :num_comp] * Sigma[:num_comp]   # [n_traits x num_comp]
 
    ideal_v_k = np.zeros(num_comp)
    for t_idx, direction in trait_map.items():
        ideal_v_k += direction * US[t_idx - 1, :]   # t_idx is 1-based
 
    norm = np.linalg.norm(ideal_v_k)
    if norm > 0:
        ideal_v_k /= norm
    return ideal_v_k, US
 
 
def build_trait_contributions(RAW_A, US, ideal_v_k, char_idx):
    """
    For one character (column char_idx in RAW_A), compute each trait's
    signed contribution to the foundation score.
 
    RAW_A     : [n_traits x n_chars]  -- RAW_A[:, char_idx] = character vector
    US        : [n_traits x num_comp] -- trait positions in scaled factor space
    ideal_v_k : [num_comp]            -- unit vector for the foundation
 
    Returns
    -------
    contrib        : [n_traits]  each trait's contribution to cosine similarity
    trait_loadings : [n_traits]  alignment of each trait with foundation (-1 to 1)
    score          : float       total foundation score
    """
    # How much each trait aligns with this foundation
    trait_loadings = np.dot(US, ideal_v_k)           # [n_traits]
    max_load = np.max(np.abs(trait_loadings))
    if max_load > 0:
        trait_loadings = trait_loadings / max_load    # normalise to [-1, 1]
 
    # Character's trait vector: axis=0 is traits, axis=1 is characters
    char_trait_vec = RAW_A[:, char_idx]              # [n_traits]  CORRECTED
 
    char_norm = np.linalg.norm(char_trait_vec)
    if char_norm == 0:
        char_norm = 1.0
 
    contrib = (char_trait_vec * trait_loadings) / char_norm
    score   = float(np.sum(contrib))
    return contrib, trait_loadings, score
 
 
# ==============================================================================
# Allotaxonometer: rank-turbulence divergence
# ==============================================================================
 
def rank_turbulence_divergence(contrib1, contrib2, trait_loadings,
                                alpha=1.0, loading_thresh=LOADING_THRESH):
    """
    Rank each trait by |contribution| within each character's profile,
    then compute rank-turbulence divergence per trait.
 
    alpha = 0  : log-rank divergence  |log(r1) - log(r2)|
    alpha > 0  : |r1^(-a) - r2^(-a)|^(1/a)
    """
    n        = len(contrib1)
    abs1     = np.abs(contrib1)
    abs2     = np.abs(contrib2)
    sig_mask = np.abs(trait_loadings) > loading_thresh
 
    sig_idx = np.where(sig_mask)[0]
    if len(sig_idx) == 0:
        raise ValueError("No traits passed the loading threshold. Lower loading_thresh.")
 
    # Dense rank within significant traits (rank 1 = highest |contrib|)
    rank1_full = np.full(n, np.nan)
    rank2_full = np.full(n, np.nan)
 
    order1 = np.argsort(-abs1[sig_mask])
    order2 = np.argsort(-abs2[sig_mask])
    r1 = np.empty(len(sig_idx)); r1[order1] = np.arange(1, len(sig_idx) + 1)
    r2 = np.empty(len(sig_idx)); r2[order2] = np.arange(1, len(sig_idx) + 1)
    rank1_full[sig_idx] = r1
    rank2_full[sig_idx] = r2
 
    with np.errstate(divide='ignore', invalid='ignore'):
        if alpha == 0:
            rdiv = np.where(
                sig_mask,
                np.abs(np.log(np.where(sig_mask, rank1_full, 1)) -
                       np.log(np.where(sig_mask, rank2_full, 1))),
                0.0
            )
        else:
            rdiv = np.where(
                sig_mask,
                np.abs(rank1_full ** (-alpha) - rank2_full ** (-alpha)) ** (1.0 / alpha),
                0.0
            )
    rdiv = np.nan_to_num(rdiv, nan=0.0)
 
    sign      = np.sign(contrib1 - contrib2)
    moral_dir = np.sign(trait_loadings)
 
    roles = []
    for i in range(n):
        if not sig_mask[i]:
            roles.append('insignificant')
        elif rdiv[i] <= SHARED_THRESH:
            roles.append('shared')
        elif sign[i] > 0:
            roles.append('char1')
        else:
            roles.append('char2')
 
    df = pd.DataFrame({
        'Contrib1':       contrib1,
        'Contrib2':       contrib2,
        'Rank1':          rank1_full,
        'Rank2':          rank2_full,
        'RankDiv':        rdiv,
        'SignedRankDiv':  rdiv * sign,
        'MoralDirection': moral_dir,
        'AbsLoading':     np.abs(trait_loadings),
        'role':           roles,
    })
 
    total_div = float(np.nansum(rdiv[sig_mask]))
    return df, total_div
 
 
# ==============================================================================
# Plotting
# ==============================================================================
 
def make_allotax_plot(results_df, trait_names, full_name1, full_name2,
                      foundation, score1, score2, total_div, alpha, out_path):
    results_df          = results_df.copy()
    results_df['Trait'] = trait_names
 
    sig        = results_df[results_df['role'] != 'insignificant'].copy()
    shared_df  = sig[sig['role'] == 'shared'].nlargest(5, 'AbsLoading')
    char1_df   = sig[sig['role'] == 'char1'].nlargest(TOP_N, 'RankDiv')
    char2_df   = sig[sig['role'] == 'char2'].nlargest(TOP_N, 'RankDiv')
 
    shared_df  = shared_df.assign(SignedRankDiv=0.0)
    plot_df    = (pd.concat([char2_df, shared_df, char1_df])
                    .reset_index(drop=True)
                    .sort_values('SignedRankDiv', ascending=True)
                    .reset_index(drop=True))
 
    fnd_color  = FOUNDATION_COLORS.get(foundation, '#555555')
 
    def bar_color(row):
        if row['role'] == 'shared':
            return SHARED_COLOR
        return MORAL_POS_COLOR if row['MoralDirection'] > 0 else MORAL_NEG_COLOR
 
    bar_colors = [bar_color(row) for _, row in plot_df.iterrows()]
 
    n_bars = len(plot_df)
    fig_h  = max(12, n_bars * 0.38 + 5)
    fig    = plt.figure(figsize=(18, fig_h))
    fig.patch.set_facecolor('#FAFAFA')
 
    gs = gridspec.GridSpec(2, 2,
                           height_ratios=[fig_h - 1.5, 1.5],
                           width_ratios=[3, 2],
                           hspace=0.06, wspace=0.35)
 
    ax_bar     = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_caption = fig.add_subplot(gs[1, :])
    for ax in [ax_bar, ax_scatter, ax_caption]:
        ax.set_facecolor('#FAFAFA')
 
    # Left: word-shift bars
    bars = ax_bar.barh(plot_df['Trait'], plot_df['SignedRankDiv'],
                       color=bar_colors, edgecolor='white', linewidth=0.5, height=0.72)
    ax_bar.axvline(0, color='#333333', linewidth=1.2, zorder=5)
    ax_bar.axvspan(-SHARED_THRESH, SHARED_THRESH,
                   color=SHARED_COLOR, alpha=0.10, zorder=0)
 
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        w = bar.get_width()
        if abs(w) < 0.001:
            continue
        offset = max(0.002, abs(w) * 0.03)
        x_pos  = w + (offset if w > 0 else -offset)
        ha     = 'left' if w > 0 else 'right'
        ax_bar.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    f'{w:+.3f}', va='center', ha=ha, fontsize=8, color='#555555')
 
    xlim   = ax_bar.get_xlim()
    box_kw = dict(boxstyle='round,pad=0.4', alpha=0.88)
    ax_bar.text(xlim[1] * 0.97, n_bars - 0.3,
                f'{full_name1}\n{foundation}: {score1:+.3f}',
                ha='right', va='top', fontsize=10, fontweight='bold',
                color='white', bbox={**box_kw, 'facecolor': fnd_color})
    ax_bar.text(xlim[0] * 0.97, n_bars - 0.3,
                f'{full_name2}\n{foundation}: {score2:+.3f}',
                ha='left', va='top', fontsize=10, fontweight='bold',
                color='white', bbox={**box_kw, 'facecolor': '#555555'})
 
    ax_bar.set_xlabel(
        f'Signed rank-turbulence divergence  (alpha={alpha})\n'
        f'<- drives {full_name2}  |  drives {full_name1} ->',
        fontsize=10
    )
    ax_bar.set_title(
        f'Moral Allotaxonometer -- {foundation}\n{full_name1}  vs  {full_name2}',
        fontsize=13, fontweight='bold'
    )
    ax_bar.xaxis.grid(True, linestyle='--', alpha=0.35, color='#aaaaaa')
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis='y', labelsize=8.5)
    ax_bar.legend(handles=[
        mpatches.Patch(color=MORAL_POS_COLOR, label='Morally positive trait'),
        mpatches.Patch(color=MORAL_NEG_COLOR, label='Morally negative trait'),
        mpatches.Patch(color=SHARED_COLOR,    label='Shared moral ground'),
    ], loc='lower right', fontsize=9, framealpha=0.85)
 
    # Right: rank-rank scatter
    scatter_df     = sig.dropna(subset=['Rank1', 'Rank2'])
    scatter_div    = scatter_df[scatter_df['role'] != 'shared']
    scatter_shared = scatter_df[scatter_df['role'] == 'shared']
 
    max_rdiv  = scatter_div['RankDiv'].max() if len(scatter_div) > 0 else 1.0
    sizes     = (scatter_div['RankDiv'] / (max_rdiv + 1e-9)) * 130 + 10
    sc_colors = [MORAL_POS_COLOR if d > 0 else MORAL_NEG_COLOR
                 for d in scatter_div['MoralDirection']]
 
    ax_scatter.scatter(scatter_div['Rank1'], scatter_div['Rank2'],
                       s=sizes, c=sc_colors, alpha=0.65,
                       edgecolors='white', linewidth=0.4, zorder=3)
    if len(scatter_shared) > 0:
        ax_scatter.scatter(scatter_shared['Rank1'], scatter_shared['Rank2'],
                           s=20, c=SHARED_COLOR, alpha=0.5,
                           edgecolors='white', linewidth=0.4, zorder=2)
 
    all_rank_vals = pd.concat([scatter_div[['Rank1','Rank2']],
                               scatter_shared[['Rank1','Rank2']]])
    max_rank = int(all_rank_vals.values.max()) if len(all_rank_vals) > 0 else 50
    ax_scatter.plot([1, max_rank], [1, max_rank],
                    color='#aaaaaa', linewidth=0.8, linestyle='--', zorder=1)
 
    top6 = scatter_div.nlargest(6, 'RankDiv')
    for i, row in top6.iterrows():
        ax_scatter.annotate(
            trait_names[i],
            xy=(row['Rank1'], row['Rank2']),
            xytext=(6, 3), textcoords='offset points',
            fontsize=7, color='#333333',
            arrowprops=dict(arrowstyle='-', color='#bbbbbb', lw=0.5)
        )
 
    ax_scatter.set_xlabel(f'Trait rank for {full_name1}', fontsize=10)
    ax_scatter.set_ylabel(f'Trait rank for {full_name2}', fontsize=10)
    ax_scatter.set_title('Rank-rank scatter\n(off-diagonal = divergence)',
                         fontsize=11, fontweight='bold')
    ax_scatter.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax_scatter.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax_scatter.set_axisbelow(True)
    for s_val, label in [(10, 'low'), (70, 'med'), (130, 'high')]:
        ax_scatter.scatter([], [], s=s_val, c='#888888', alpha=0.6, label=label)
    ax_scatter.legend(title='Rank divergence', fontsize=8,
                      title_fontsize=8, loc='upper left', framealpha=0.8)
 
    # Bottom: total divergence callout
    ax_caption.axis('off')
    ax_caption.text(
        0.5, 0.5,
        (f'Total {foundation} divergence  tau = {total_div:.3f}   (alpha={alpha})   |   '
         f'{len(char1_df)} traits drive {full_name1}   |   '
         f'{len(char2_df)} traits drive {full_name2}   |   '
         f'{len(shared_df)} shared traits'),
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='white', transform=ax_caption.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=fnd_color, alpha=0.85)
    )
 
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_path}")
    return total_div
 
 
def make_summary_divergence_bar(divergence_scores, full_name1, full_name2, out_path):
    """Bar chart: total tau per foundation."""
    foundations = list(divergence_scores.keys())
    scores      = [divergence_scores[f] for f in foundations]
    colors      = [FOUNDATION_COLORS.get(f, '#888888') for f in foundations]
 
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
 
    bars = ax.bar(foundations, scores, color=colors,
                  edgecolor='white', linewidth=0.8, width=0.6)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(scores) * 0.015,
                f'{score:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#333333')
 
    ax.set_ylabel('Total rank-turbulence divergence  tau', fontsize=12)
    ax.set_title(
        f'Which moral foundation separates them most?\n{full_name1}  vs  {full_name2}',
        fontsize=14, fontweight='bold'
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)
 
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Summary saved -> {out_path}")
 
 
# ==============================================================================
# Main
# ==============================================================================
 
def main():
    parser = argparse.ArgumentParser(description='Moral Allotaxonometer')
    parser.add_argument('--char1',      type=str,   required=True)
    parser.add_argument('--char2',      type=str,   required=True)
    parser.add_argument('--foundation', type=str,   default='Fairness',
                        help='Foundation name or "all"')
    parser.add_argument('--alpha',      type=float, default=1.0,
                        help='Turbulence parameter (0=log-rank, 1=default)')
    args = parser.parse_args()
 
    try:
        traits_df = pd.read_csv(os.path.join(data_path, 'data_traits.tsv'),      sep='\t')
        char_df   = pd.read_csv(os.path.join(data_path, 'data_characters.tsv'),  sep='\t')
        RAW_A     = load_matrix('data_raw_A.txt')                  # [464 x 2000]
        U         = load_matrix('data_archetype_space_U.txt')      # [464 x 464]
        Sigma     = load_matrix('data_archetype_space_Sigma.txt')   # [464]
    except Exception as e:
        print(f"Error loading data: {e}")
        return
 
    # Validate orientation up front so errors are caught immediately
    n_traits, n_chars = RAW_A.shape
    print(f"RAW_A shape: {RAW_A.shape}  (expect [464, 2000])")
    assert n_traits == 464,  f"Expected 464 traits on axis 0, got {n_traits}"
    assert n_chars  == 2000, f"Expected 2000 chars on axis 1, got {n_chars}"
 
    from config import moral_foundations_config as config
    num_comp    = config.NUM_COMPONENTS
    trait_names = traits_df['differential'].values
 
    idx1, name1 = find_char(char_df, args.char1)
    idx2, name2 = find_char(char_df, args.char2)
    print(f"Character 1: {name1} (col {idx1})")
    print(f"Character 2: {name2} (col {idx2})")
 
    foundations = list(FOUNDATION_COLORS.keys()) if args.foundation == 'all' \
                  else [args.foundation]
 
    divergence_scores = {}
 
    for foundation in foundations:
        trait_map = config.FOUNDATIONS.get(foundation)
        if not trait_map:
            print(f"  Skipping '{foundation}' -- not in config.")
            continue
 
        print(f"\n-- {foundation} --")
 
        # Build foundation vector once per foundation (shared across characters)
        ideal_v_k, US = build_foundation_vector(U, Sigma, trait_map, num_comp)
 
        contrib1, loadings, score1 = build_trait_contributions(
            RAW_A, US, ideal_v_k, idx1)
        contrib2, _,        score2 = build_trait_contributions(
            RAW_A, US, ideal_v_k, idx2)
 
        print(f"  {name1} score: {score1:+.4f}")
        print(f"  {name2} score: {score2:+.4f}")
 
        results_df, total_div = rank_turbulence_divergence(
            contrib1, contrib2, loadings, alpha=args.alpha)
 
        print(f"  Total divergence tau = {total_div:.4f}")
        divergence_scores[foundation] = total_div
 
        fname    = (f'allotax_{name1}_{name2}_{foundation}_a{args.alpha}.png'
                    .replace(' ', '_'))
        out_path = os.path.join(figs_path, fname)
 
        make_allotax_plot(
            results_df, trait_names, name1, name2,
            foundation, score1, score2, total_div, args.alpha, out_path
        )
 
    if args.foundation == 'all' and len(divergence_scores) > 1:
        summary_path = os.path.join(
            figs_path,
            f'allotax_summary_{name1}_{name2}.png'.replace(' ', '_')
        )
        make_summary_divergence_bar(divergence_scores, name1, name2, summary_path)
 
        print("\n-- Divergence ranking (most -> least) --")
        for f, d in sorted(divergence_scores.items(), key=lambda x: -x[1]):
            print(f"  {f:12s}  tau = {d:.4f}")
 
 
if __name__ == "__main__":
    main()