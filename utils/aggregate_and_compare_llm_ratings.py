import pandas as pd
import os
from pathlib import Path

# Config
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "config" / "llm_rating" / "outputs"
FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "purity", "liberty", "general"]

def aggregate_provider(provider_path):
    batches = list(provider_path.glob("batch_*.tsv"))
    if len(batches) < 19:
        print(f"Skipping {provider_path.name}: Only {len(batches)}/19 batches found.")
        return None
    
    print(f"Combining 19 batches for {provider_path.name}...")
    df_list = []
    for b in sorted(batches):
        try:
            df_list.append(pd.read_csv(b, sep='\t'))
        except Exception as e:
            print(f"Error reading {b.name}: {e}")
            
    if not df_list:
        return None
        
    combined = pd.concat(df_list).sort_values("trait_index")
    # Drop duplicates just in case
    combined = combined.drop_duplicates(subset=["trait_index"])
    
    output_file = provider_path / "combined_ratings.tsv"
    combined.to_csv(output_file, sep='\t', index=False)
    return combined

def compare_models(model_dfs):
    # Get all unique trait indices from all models
    all_indices = set()
    for df in model_dfs.values():
        all_indices.update(df['trait_index'].tolist())
    all_indices = sorted(list(all_indices))
    
    comparison_rows = []

    for idx in all_indices:
        # Get row for this trait from each model
        model_data = {}
        differential = "Unknown"
        
        for name, df in model_dfs.items():
            subset = df[df['trait_index'] == idx]
            if not subset.empty:
                model_data[name] = subset.iloc[0]
                differential = subset.iloc[0]['differential']
        
        if not model_data:
            continue
            
        agreed = []
        divergent = []
        
        for f in FOUNDATIONS:
            # Check ratings for this foundation across all models that have data for this trait
            ratings = {name: data[f] for name, data in model_data.items() if f in data}
            if not ratings:
                continue
                
            ones = [name for name, r in ratings.items() if str(r) in ('1', '1.0')]
            zeros = [name for name, r in ratings.items() if str(r) in ('0', '0.0')]
            
            # Agreement: Everyone said 1
            if len(ones) == len(ratings) and len(ones) > 0:
                agreed.append(f)
            # Divergence: Some said 1, some said 0
            elif len(ones) > 0 and len(zeros) > 0:
                divergent.append(f"{f} ({', '.join(ones)} only)")
        
        # Only add row if there is at least one +1 or one divergence
        if agreed or divergent:
            comparison_rows.append({
                "trait_index": idx,
                "differential": differential,
                "agreed foundation": ", ".join(agreed),
                "divergence": ", ".join(divergent)
            })
            
    return pd.DataFrame(comparison_rows)

def main():
    model_dataframes = {}
    
    if not OUTPUT_DIR.exists():
        print(f"Output directory not found: {OUTPUT_DIR}")
        return

    #Aggregate individuals
    for provider_dir in OUTPUT_DIR.iterdir():
        if provider_dir.is_dir():
            df = aggregate_provider(provider_dir)
            if df is not None:
                model_dataframes[provider_dir.name] = df
                
    #Compare if we have at least 2 models
    if len(model_dataframes) >= 2:
        print(f"\nComparing {len(model_dataframes)} models: {', '.join(model_dataframes.keys())}")
        comparison_df = compare_models(model_dataframes)
        summary_path = OUTPUT_DIR / "llm_comparison_agreement.tsv"
        comparison_df.to_csv(summary_path, sep='\t', index=False)
        print(f"Success! Comparison summary written to {summary_path}")
    else:
        print("\nNot enough completed models to perform comparison. Need at least 2.")

if __name__ == "__main__":
    main()
