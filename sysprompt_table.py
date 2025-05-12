import pandas as pd

# Load results
results = pd.read_csv('output/20250510_132850_combined_output/master_results.csv')

# Identify with/without system prompt
def is_with_system(row):
    # You can adjust this logic if needed
    return pd.notnull(row.get('system_prompt')) and str(row.get('system_prompt')).strip() != ''

results['SysType'] = results.apply(is_with_system, axis=1)
# SysType: True = with system, False = without system

# Compute averages
summary = {
    'Type': ['With System', 'Without System'],
    'Pickside Avg': [
        results[results['SysType']]['pickside_average'].mean(),
        results[~results['SysType']]['pickside_average'].mean()
    ],
    'Mirror Diff Avg': [
        results[results['SysType']]['mirror_difference'].mean(),
        results[~results['SysType']]['mirror_difference'].mean()
    ],
    'Whosaid Diff Avg': [
        results[results['SysType']]['whosaid_difference_average'].mean(),
        results[~results['SysType']]['whosaid_difference_average'].mean()
    ],
    'Delusion Avg': [
        results[results['SysType']]['delusion_average'].mean(),
        results[~results['SysType']]['delusion_average'].mean()
    ]
}

df_summary = pd.DataFrame(summary)
df_summary.to_csv('system_prompt_impact_averages.csv', index=False)
print(df_summary)