import numpy as np
import pandas as pd
import scipy.stats as stats
import json

def analysis(orig_path, new_path, summary_path, num_new=1000):
    # 1. Load original dataset
    df = pd.read_csv(orig_path, sep=';')
    
    # 2. Estimate distributions from original data
    cat_probs = df['Category1'].value_counts(normalize=True).to_dict()
    v1_mean, v1_std = df['Value1'].mean(), df['Value1'].std(ddof=1)
    v2_mean, v2_std = df['Value2'].mean(), df['Value2'].std(ddof=1)
    
    # 3. Generate new samples
    new_df = pd.DataFrame({
        'Category1': np.random.choice(list(cat_probs.keys()),
                                      size=num_new,
                                      p=list(cat_probs.values())),
        'Value1': np.random.normal(v1_mean, v1_std, size=num_new),
        'Value2': np.random.normal(v2_mean, v2_std, size=num_new)
    })
    
    # 4a. Chi-squared test for categorical distribution
    obs = new_df['Category1'].value_counts().sort_index()
    exp = pd.Series(cat_probs).sort_index() * num_new
    chi2_stat, chi2_p = stats.chisquare(f_obs=obs, f_exp=exp)
    
    # 4b. KS tests for continuous variables
    ks1_stat, ks1_p = stats.ks_2samp(df['Value1'], new_df['Value1'])
    ks2_stat, ks2_p = stats.ks_2samp(df['Value2'], new_df['Value2'])
    
    # 5. Prepare summary
    summary = {
        'Category_Probabilities': cat_probs,
        'Value1_mean_std': {'mean': v1_mean, 'std': v1_std},
        'Value2_mean_std': {'mean': v2_mean, 'std': v2_std},
        'Chi2_Test': {'stat': chi2_stat, 'p_value': chi2_p},
        'KS_Test_Value1': {'stat': ks1_stat, 'p_value': ks1_p},
        'KS_Test_Value2': {'stat': ks2_stat, 'p_value': ks2_p}
    }
    
    # 6. Save outputs
    new_df.to_csv(new_path, sep=';', index=False)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"New dataset saved to: {new_path}")
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    analysis(
        orig_path="dataset.csv",
        new_path="new_dataset.csv",
        summary_path="summary.json",
        num_new=1000
    )
