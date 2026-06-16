import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ──────────────────────────────────────────────────────────────────────
data_path    = 'data/plain/current/2000/'
figs_path    = 'figs/'
os.makedirs(figs_path, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────
CHAR1_COLOR = '#1f77b4' # Blue
CHAR2_COLOR = '#d62728' # Red
SEED_ZONE_COLOR = '#FFF9C4' # Light Gold
SEED_BORDER_COLOR = '#FBC02D' # Gold Border

def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))

def find_char(char_df, name):
    matches = char_df[char_df['character'].str.contains(name, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"Character '{name}' not found.")
    return matches.index[0], matches.iloc[0]['character']

def get_trait_contributions(RAW_A, U, V, trait_map, num_comp, idx):
    U_k = U[:, :num_comp]
    foundation_v = np.zeros(num_comp)
    for t_idx, direction in trait_map.items():
        foundation_v += direction * U_k[t_idx - 1, :]
    
    fn_norm = np.linalg.norm(foundation_v)
    if fn_norm == 0: fn_norm = 1.0

    char_trait_vec = RAW_A[:, idx]
    char_k_vec = np.dot(U_k.T, char_trait_vec)
    ch_norm = np.linalg.norm(char_k_vec)
    if ch_norm == 0: ch_norm = 1.0

    trait_loadings = np.dot(U_k, foundation_v) / fn_norm
    contrib = (char_trait_vec * trait_loadings) / ch_norm
    
    return contrib, trait_loadings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char1', type=str, required=True)
    parser.add_argument('--char2', type=str, required=True)
    parser.add_argument('--foundation', type=str, default='Fairness')
    args = parser.parse_args()

    traits_df = pd.read_csv(os.path.join(data_path, 'data_traits.tsv'), sep='\t')
    char_df   = pd.read_csv(os.path.join(data_path, 'data_characters.tsv'), sep='\t')
    RAW_A     = load_matrix('data_raw_A.txt')
    U         = load_matrix('data_archetype_space_U.txt')
    V         = load_matrix('data_archetype_space_V.txt')

    from config import moral_foundations_config as config
    trait_map = config.FOUNDATIONS.get(args.foundation)
    if not trait_map: return

    idx1, name1 = find_char(char_df, args.char1)
    idx2, name2 = find_char(char_df, args.char2)

    c1, loadings = get_trait_contributions(RAW_A, U, V, trait_map, config.NUM_COMPONENTS, idx1)
    c2, _        = get_trait_contributions(RAW_A, U, V, trait_map, config.NUM_COMPONENTS, idx2)

    moral_poles = [traits_df.iloc[i]['right pole'] if loadings[i] >= 0 else traits_df.iloc[i]['left pole'] for i in range(len(loadings))]

    results = pd.DataFrame({
        'Trait': traits_df['differential'],
        'MoralPole': moral_poles,
        'config_idx': traits_df['index'],
        'Contrib1': c1,
        'Contrib2': c2,
        'AbsDiff': np.abs(c1 - c2),
        'Loading': np.abs(loadings)
    })

    # ── Selection Logic (Back to what you liked) ──────────────────────────────
    seed_indices = list(trait_map.keys())
    seeds_df = results[results['config_idx'].isin(seed_indices)].copy()
    seeds_df['Type'] = 'B: Seed'

    # Filter for significant loadings, then take top 12 differences
    discovered_df = results[~results['config_idx'].isin(seed_indices)].copy()
    significant_discovered = discovered_df[discovered_df['Loading'] > 0.1].nlargest(12, 'AbsDiff')
    significant_discovered['Type'] = 'A: Discovered'

    # Combine and sort so Seeds (B) are at the END (Top of plot)
    plot_df = pd.concat([significant_discovered, seeds_df]).sort_values(['Type', 'AbsDiff'], ascending=[True, True])

    # ── Plotting ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    
    y = np.arange(len(plot_df))
    height = 0.38

    ax.barh(y - height/2, plot_df['Contrib1'], height, label=name1, color=CHAR1_COLOR, edgecolor='white', alpha=0.9, zorder=3)
    ax.barh(y + height/2, plot_df['Contrib2'], height, label=name2, color=CHAR2_COLOR, edgecolor='white', alpha=0.9, zorder=3)

    labels = [f"{row['Trait']} → ({row['MoralPole']})" for _, row in plot_df.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, fontweight='bold')
    
    # ── Seed Highlight ────────────────────────────────────────────────────────
    num_seeds = len(seeds_df)
    # Highlight the last 'num_seeds' items (the top of the plot)
    ax.axhspan(len(plot_df) - num_seeds - 0.5, len(plot_df) - 0.5, 
               color=SEED_ZONE_COLOR, alpha=0.5, label='Config traits', zorder=1)
    ax.axhline(len(plot_df) - num_seeds - 0.5, color=SEED_BORDER_COLOR, linewidth=2, linestyle='--', zorder=2)

    ax.legend(fontsize=12, loc='lower right', frameon=True)
    ax.axvline(0, color='black', linewidth=1.5, alpha=0.7, zorder=4)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    
    plt.title(f'Moral divergence: {name1} vs {name2}\nFoundation: {args.foundation}', size=22, fontweight='bold', pad=25)
    plt.xlabel('Individual trait contribution to moral Score', size=14)

    # Total Score Summary
    s1, s2 = np.sum(c1), np.sum(c2)
    plt.text(0.02, 0.98, f'{name1} Total: {s1:.2f}\n{name2} Total: {s2:.2f}', 
             transform=ax.transAxes, ha='left', va='top', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CCCCCC'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(figs_path, f'moral_comparison_{name1}_{name2}_{args.foundation}.png'.replace(" ","_"))
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"SUCCESS: saved → {out_path}")

if __name__ == "__main__":
    main()
