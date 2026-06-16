import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# Set paths
results_path = 'results/'
figs_path = 'figs/'
input_file = os.path.join(results_path, 'robust_moral_analysis_results.csv')

# Ensure figs directory exists
if not os.path.exists(figs_path):
    os.makedirs(figs_path)

def create_radar_chart(characters_data, character_names, output_filename):
    """
    Creates a radar chart for one or more characters using Foundation_Robust_Avg scores.
    """
    labels = ['Care', 'Fairness', 'Loyalty', 'Authority', 'Sanctity', 'Liberty']
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'] # Green, Red, Blue, Orange

    for i, data in enumerate(characters_data):
        values = data + data[:1] # Close the loop
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=3, label=character_names[i])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.15)

    # Styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set axis labels
    plt.xticks(angles[:-1], labels, color='black', size=14, fontweight='bold')

    # Set radial limits and labels
    # Cosine similarity ranges from -1 to 1. 
    # We'll set the limit to show the full spectrum.
    ax.set_ylim(-0.5, 1.0) 
    ax.set_rlabel_position(0)
    # Add a dashed line at 0 to show the "Neutral" boundary
    grid_angles = np.linspace(0, 2*np.pi, 100)
    ax.plot(grid_angles, np.zeros(100), color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    plt.yticks([-0.4, -0.2, 0.2, 0.4, 0.6, 0.8], ["-0.4", "-0.2", "0.2", "0.4", "0.6", "0.8"], color="grey", size=10)

    plt.title('Stable Moral Fingerprint (Robust Avg)', size=22, color='black', y=1.1, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(figs_path, output_filename)
    plt.savefig(save_path, dpi=300) # High DPI for posters
    print(f"SUCCESS: Radar chart saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate Moral Radar Charts')
    parser.add_argument('--char1', type=str, help='Name of the first character')
    parser.add_argument('--char2', type=str, help='Name of the second character (optional)')
    args = parser.parse_args()

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run generate_robust_moral_scores.py first.")
        return

    df = pd.read_csv(input_file)
    
    # Target columns
    foundations = ['Care_Robust_Avg', 'Fairness_Robust_Avg', 'Loyalty_Robust_Avg', 
                   'Authority_Robust_Avg', 'Sanctity_Robust_Avg', 'Liberty_Robust_Avg']

    def get_char_data(name):
        # Case insensitive search
        row = df[df['Character'].str.lower() == name.lower()]
        if row.empty:
            # Try fuzzy match if exact fails
            matches = df[df['Character'].str.contains(name, case=False)]
            if not matches.empty:
                row = matches.iloc[[0]]
                print(f"Exact match not found. Using best match: {row['Character'].values[0]}")
            else:
                print(f"Character '{name}' not found in results.")
                return None
        return row[foundations].values.flatten().tolist(), row['Character'].values[0]

    if not args.char1:
        print("Please provide at least one character name.")
        print("Top characters in dataset:")
        print(df.sort_values('Robust_Avg', ascending=False)[['Character', 'Story']].head(10))
        return

    data1, name1 = get_char_data(args.char1)
    if data1 is None: return

    chars_data = [data1]
    names = [name1]
    out_name = f"radar_{name1.replace(' ', '_')}"

    if args.char2:
        data2, name2 = get_char_data(args.char2)
        if data2:
            chars_data.append(data2)
            names.append(name2)
            out_name += f"_vs_{name2.replace(' ', '_')}"

    create_radar_chart(chars_data, names, f"{out_name}.png")

if __name__ == "__main__":
    main()
