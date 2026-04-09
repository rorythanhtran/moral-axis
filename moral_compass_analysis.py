import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_detailed_moral_compass():
    print("--- Starting Detailed Character Moral Compass Analysis ---")
    
    # 1. LOAD DATA
    data_dir = os.path.join('data', 'plain', 'current', '2000')
    characters_df = pd.read_csv(os.path.join(data_dir, 'data_characters.tsv'), sep='\t')
    traits_df = pd.read_csv(os.path.join(data_dir, 'data_traits.tsv'), sep='\t')
    A_raw = np.loadtxt(os.path.join(data_dir, 'data_raw_A.txt'))
    A = A_raw.T 

    # 2. SELECT 4 MORAL TRAITS
    # We use: Honorable, Loyal, Heroic, Forgiving
    indices = [13, 27, 38, 24]
    trait_names = [traits_df.iloc[i]['left pole'] for i in indices]
    
    # Flip signs so 'Good' is positive
    A_moral = -A[:, indices]

    # 3. PERFORM SVD
    U, S, Vt = np.linalg.svd(A_moral, full_matrices=False)
    PCs = U * S # Coordinates for the plot

    # 4. ANALYZE LOADINGS (What do the axes mean?)
    # Vt[0] is the loading for PC1 (X-axis)
    # Vt[1] is the loading for PC2 (Y-axis)
    
    print("\n--- Axis Loading Analysis ---")
    loadings_df = pd.DataFrame(Vt[:2], columns=trait_names, index=['PC1 (X)', 'PC2 (Y)'])
    print(loadings_df.round(3))
    
    # Identify the "Flavor" of PC2
    # Usually PC2 contrasts two traits. We'll find the max and min loadings.
    pc2_max_trait = trait_names[np.argmax(Vt[1])]
    pc2_min_trait = trait_names[np.argmin(Vt[1])]

    # 5. SCALE EXPLANATION
    # Input data is [-0.5, 0.5]. 
    # PC scores are sums of these weighted by singular values.
    # Max possible PC1 score is approx 0.5 * sum(Vt[0]) * sqrt(N_traits)
    
    # 6. VISUALIZATION
    plt.figure(figsize=(12, 10))
    plt.axhline(0, color='black', lw=1, alpha=0.3)
    plt.axvline(0, color='black', lw=1, alpha=0.3)
    
    # Plot background
    plt.scatter(PCs[:, 0], PCs[:, 1], alpha=0.15, s=12, c='gray', label='All 2000 Characters')

    # Highlight specific characters
    highlights = {
        'Samwise Gamgee': 'blue',
        'The Joker': 'black',
        'Atticus Finch': 'green',
        'Lord Voldemort': 'purple',
        'SpongeBob SquarePants': 'gold',
        'Batman': 'navy',
        'Ned Stark': 'brown',
        'Forrest Gump': 'red', 
        'Sam Obisanya':'orange', 
        'Aang': 'pink'
    }

    print("\n--- Character Positions ---")
    for char, color in highlights.items():
        try:
            idx = characters_df[characters_df['character'] == char].index[0]
            x, y = PCs[idx, 0], PCs[idx, 1]
            plt.scatter(x, y, color=color, s=80, edgecolors='white', zorder=5)
            plt.annotate(char, (x, y), xytext=(5, 5), textcoords='offset points', 
                         fontsize=10, fontweight='bold', color=color)
            print(f"- {char:22}: X={x: 6.2f} (Virtue), Y={y: 6.2f} (Type)")
        except:
            continue

    # 7. AXIS ANNOTATIONS
    plt.xlabel(f'X-Axis: Moral Virtue (combination of all 4 traits)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Y-Axis: Moral Conflict ({pc2_max_trait} vs {pc2_min_trait})', fontsize=12, fontweight='bold')
    
    # Quadrant Labels based on PC2 analysis
    plt.text(0.5, 0.8, f'POSITIVE VIRTUE\n({pc2_max_trait} Lean)', transform=plt.gca().transAxes, 
             fontsize=12, color='darkgreen', alpha=0.6, fontweight='bold', ha='center')
    plt.text(0.5, 0.1, f'NEGATIVE VIRTUE\n({pc2_min_trait} Lean)', transform=plt.gca().transAxes, 
             fontsize=12, color='darkred', alpha=0.6, fontweight='bold', ha='center')

    plt.title('The SVD moral compass', fontsize=15, pad=20)
    plt.grid(alpha=0.1)
    
    # Save results
    plt.savefig('detailed_moral_compass.png', dpi=300, bbox_inches='tight')
    print("\nSuccess! Detailed plot saved as 'detailed_moral_compass.png'")
    plt.show()

if __name__ == "__main__":
    generate_detailed_moral_compass()
