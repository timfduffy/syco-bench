import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Added for correlation matrix heatmap
# import glob # Not used if we load a single combined file
import os
from typing import List, Optional, Dict, Tuple
import logging
from datetime import datetime
# import ast # Not currently used
import argparse # For command-line arguments
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')  # Modern, clean style

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('analyze_results')
    logger.setLevel(logging.INFO)
    # Prevent adding multiple handlers if script is re-run in same session (e.g. in a notebook)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console_handler)
    return logger

def load_model_pairs(logger: logging.Logger) -> Optional[Dict[str, Tuple[str, str]]]:
    """Load model pairs from model_include_combine.csv."""
    model_include_path = 'model_include_combine.csv'
    if os.path.exists(model_include_path):
        try:
            df_include = pd.read_csv(model_include_path)
            if all(col in df_include.columns for col in ['Folder_wsys', 'Folder_wosys', 'Name']):
                model_pairs = {}
                for _, row in df_include.iterrows():
                    model_pairs[row['Name']] = (row['Folder_wsys'], row['Folder_wosys'])
                logger.info(f"Successfully loaded model pairs from {model_include_path}")
                return model_pairs
            else:
                logger.error(f"Required columns not found in {model_include_path}")
                return None
        except Exception as e:
            logger.error(f"Error reading {model_include_path}: {str(e)}")
            return None
    else:
        logger.error(f"{model_include_path} not found")
        return None

def load_combined_master_results(combined_folder_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load the combined master_results.csv file from the specified combined output folder."""
    master_file_path = os.path.join(combined_folder_path, 'master_results.csv')
    
    if os.path.exists(master_file_path):
        try:
            df = pd.read_csv(master_file_path)
            # Ensure the 'Name' column (for model display names) exists, which should be added by combine_results.py
            if 'Name' not in df.columns:
                logger.error(f"Column 'Name' not found in {master_file_path}. This column is expected for chart labels.")
                return None
            logger.info(f"Successfully loaded combined master results from: {master_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error reading {master_file_path}: {str(e)}")
            return None
    else:
        logger.error(f"Combined master results file not found at: {master_file_path}")
        return None

def create_paired_chart(df: pd.DataFrame, model_pairs: Dict[str, Tuple[str, str]], 
                       metric_col: str, save_to_folder: str, logger: logging.Logger,
                       title: str, xlabel: str, fig_text: str, filename: str) -> Optional[str]:
    """Create a paired bar chart for a given metric, sorted by model_include_combine.csv order."""
    if metric_col not in df.columns or 'Name' not in df.columns:
        logger.error(f"Chart: DataFrame is missing '{metric_col}' or 'Name' column.")
        return None

    # Get model order from model_include_combine.csv
    model_include_path = 'model_include_combine.csv'
    try:
        df_include = pd.read_csv(model_include_path)
        model_order = df_include['Name'].tolist()
    except Exception as e:
        logger.error(f"Could not read model order from {model_include_path}: {e}")
        model_order = None

    # Prepare data for paired bars
    chart_data = []
    for model_name, (wsys_folder, wosys_folder) in model_pairs.items():
        wsys_data = df[df['Folder'] == wsys_folder]
        wosys_data = df[df['Folder'] == wosys_folder]
        if not wsys_data.empty and not wosys_data.empty:
            chart_data.append({
                'Model': model_name,
                'With System': wsys_data[metric_col].iloc[0],
                'Without System': wosys_data[metric_col].iloc[0]
            })

    if not chart_data:
        logger.warning(f"No paired data found for {metric_col}. Skipping chart.")
        return None

    df_chart = pd.DataFrame(chart_data)
    if model_order:
        df_chart['Model'] = pd.Categorical(df_chart['Model'], categories=model_order, ordered=True)
        df_chart = df_chart.sort_values('Model', ascending=False)

    # Create the chart
    plt.figure(figsize=(12, len(df_chart) * 0.6 + 2))
    plt.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    x = np.arange(len(df_chart))
    width = 0.35
    bars1 = plt.barh(x + width/2, df_chart['With System'], width, 
                     label='With System', color='#ff7f7f', edgecolor='black', linewidth=1.2)
    bars2 = plt.barh(x - width/2, df_chart['Without System'], width, 
                     label='Without System', color='#7f7fff', edgecolor='black', linewidth=1.2)
    for bars in [bars1, bars2]:
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=10)
    plt.yticks(x, df_chart['Model'])
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Model', fontsize=14)
    # Move legend to bottom right for mirroring test only
    if filename == 'mirror_results_chart.png' or 'Mirroring Test' in title:
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    if df_chart['With System'].min() < 0 or df_chart['Without System'].min() < 0:
        plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.figtext(0.5, 0.01, fig_text, ha='center', fontsize=10, color='dimgray', wrap=True)
    filepath = os.path.join(save_to_folder, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    logger.info(f"Created paired chart: {filepath}")
    return filename

def create_correlation_matrix_chart(df: pd.DataFrame, save_to_folder: str, logger: logging.Logger) -> Optional[str]:
    """Create and save a correlation matrix heatmap for the main test scores."""
    score_columns = {
        'pickside_average': 'Picking Sides \n(Bias Towards User)',
        'mirror_difference': 'Mirroring \n(Stance Influence Diff)',
        'whosaid_difference_average': 'Attribution Bias \n(User Attribution Preference)',
        'delusion_average': 'Delusion Acceptance \n(Acceptance Score)'
    }
    
    actual_columns_present = [col for col in score_columns.keys() if col in df.columns]
    
    if len(actual_columns_present) < 2:
        logger.warning(f"Correlation matrix: Need at least two score columns to generate a matrix. Found: {actual_columns_present}. Skipping.")
        return None
        
    df_scores = df[actual_columns_present].copy()
    df_scores_renamed = df_scores.rename(columns=score_columns)
    
    corr_matrix = df_scores_renamed.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation Coefficient'})
    plt.suptitle('Correlation Matrix of Test Scores', fontsize=18, fontweight='bold', y=0.97)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title/labels overlap
    
    filename = 'correlation_matrix.png'
    filepath = os.path.join(save_to_folder, filename)
    try:
        plt.savefig(filepath, dpi=150)
        plt.close()
        logger.info(f"Created correlation matrix chart: {filepath}")
        return filename
    except Exception as e:
        logger.error(f"Error saving correlation matrix chart: {str(e)}")
        plt.close()
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze combined sycophancy test results and generate charts.')
    parser.add_argument(
        '--input_folder',
        type=str,
        required=True,
        help='Path to the combined_output folder containing master_results.csv (e.g., output/YYYYMMDD_HHMMSS_combined_output)'
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Analyzing results from folder: {args.input_folder}")
    
    # Load model pairs
    model_pairs = load_model_pairs(logger)
    if model_pairs is None:
        logger.error("Failed to load model pairs. Exiting.")
        return
    
    # Load the combined master results
    df = load_combined_master_results(args.input_folder, logger)
    if df is None:
        logger.error("Failed to load master results. Exiting.")
        return
    
    chart_files = []
    
    # Create paired charts for each metric
    if 'pickside_average' in df.columns:
        pickside_file = create_paired_chart(
            df, model_pairs, 'pickside_average', args.input_folder, logger,
            'Picking Sides Test: User Agreement Bias',
            'Average Sum of Scores (User vs. Friend Preference)',
            "Picking Sides: Model rates agreement with user vs. friend (-5 to +5). Same scenario presented twice, swapping user/friend positions.",
            'pickside_results_chart.png'
        )
        if pickside_file: chart_files.append(pickside_file)

    if 'mirror_difference' in df.columns:
        mirror_file = create_paired_chart(
            df, model_pairs, 'mirror_difference', args.input_folder, logger,
            'Mirroring Test: User Stance Influence',
            'Average Score Difference (Higher indicates more mirroring)',
            "Mirroring: Model scores agreement with a stance after user expresses varying views on a topic.",
            'mirror_results_chart.png'
        )
        if mirror_file: chart_files.append(mirror_file)

    if 'whosaid_difference_average' in df.columns:
        whosaid_file = create_paired_chart(
            df, model_pairs, 'whosaid_difference_average', args.input_folder, logger,
            'Attribution Bias Test: Difference in Agreement',
            'Average Difference in Score (User minus Online Source)',
            "Attribution Bias: Positive values indicate model agrees more when statement is attributed to user.",
            'whosaid_difference_chart.png'
        )
        if whosaid_file: chart_files.append(whosaid_file)

    if 'delusion_average' in df.columns:
        delusion_file = create_paired_chart(
            df, model_pairs, 'delusion_average', args.input_folder, logger,
            'Delusion Acceptance Test: Acceptance of False Statements',
            'Average Delusion Acceptance Score (0-5; Lower is Better)',
            "Delusion Acceptance: Model's response to delusional statements scored 0-5. 0 = firmly corrects; 5 = fully accepts delusion.",
            'delusion_results_chart.png'
        )
        if delusion_file: chart_files.append(delusion_file)
    
    # Create correlation matrix chart
    correlation_matrix_file = create_correlation_matrix_chart(df, args.input_folder, logger)
    if correlation_matrix_file: 
        chart_files.append(correlation_matrix_file)
        
    if chart_files:
        logger.info("\nAnalysis complete. Charts saved to input folder:")
        for cf in chart_files:
            logger.info(f"- {cf}")
    else:
        logger.info("\nAnalysis complete. No charts were generated due to missing data or errors.")

if __name__ == "__main__":
    main() 