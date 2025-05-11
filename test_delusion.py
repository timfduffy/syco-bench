import csv
import logging
import os
from typing import List, Dict
from model_utils import (
    get_model_response, 
    get_judge_scores, 
    get_median_score, 
    get_output_dir,
    setup_test_args,
    load_test_data_csv,
    save_results
)
import statistics
import math

def process_delusions(delusion_statements: List[str], model: str, logger: logging.Logger, system_prompt: str = None) -> List[Dict]:
    """Process delusional statements and get model responses. Returns list of result dicts."""
    # Process each statement
    results = []
    for i, statement in enumerate(delusion_statements):
        logger.info(f"\nProcessing statement {i+1}/{len(delusion_statements)}: {statement[:100]}...")
        
        # Get model response
        response = get_model_response(statement, model=model)
        
        # Get judge scores
        scores = get_judge_scores(
            statement,
            response,
            'delusion',
            statement=statement
        )
        
        # Calculate median score
        median_score = get_median_score(scores)
        
        # Store results
        results.append({
            'statement': statement,
            'response': response,
            'judge1_score': scores[0],
            'judge2_score': scores[1],
            'judge3_score': scores[2],
            'median_score': median_score
        })
        
        logger.info(f"Judge scores: {scores}")
        logger.info(f"Median score: {median_score}")
    
    return results

def main():
    """Main function to run the delusion test."""
    args, system_prompt_content = setup_test_args(
        test_description='Test model responses to delusional statements', 
        model_required=True
    )

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
        logger.propagate = False
    
    logger.info(f"Starting delusion test with model: {args.model}")
    
    # Load delusions from CSV
    loaded_data = load_test_data_csv(
        csv_filename='questions_delusion.csv',
        required_columns=['statement'],
        encoding='utf-8-sig'
    )

    if not loaded_data:
        logger.error("Exiting: Failed to load or no valid data found in questions_delusion.csv")
        return

    # Extract statements
    all_statements = [item['statement'] for item in loaded_data]

    # Apply limit if specified
    if args.limit:
        statements_to_process = all_statements[:args.limit]
        logger.info(f"Limited to {len(statements_to_process)} statements")
    else:
        statements_to_process = all_statements
        logger.info(f"Loaded {len(statements_to_process)} statements")

    if not statements_to_process:
        logger.error("No statements to process after applying limit or loading.")
        return

    try:
        # Process all statements
        results = process_delusions(statements_to_process, args.model, logger, system_prompt_content)

        # Calculate average scores from the results returned by process_delusions
        valid_results_median = [r['median_score'] for r in results if r['median_score'] is not None and not math.isnan(r['median_score'])]
        average_median = statistics.mean(valid_results_median) if valid_results_median else float('nan')
        
        valid_results_j1 = [r['judge1_score'] for r in results if r['judge1_score'] is not None and not math.isnan(r['judge1_score'])]
        average_judge1 = statistics.mean(valid_results_j1) if valid_results_j1 else float('nan')

        valid_results_j2 = [r['judge2_score'] for r in results if r['judge2_score'] is not None and not math.isnan(r['judge2_score'])]
        average_judge2 = statistics.mean(valid_results_j2) if valid_results_j2 else float('nan')

        valid_results_j3 = [r['judge3_score'] for r in results if r['judge3_score'] is not None and not math.isnan(r['judge3_score'])]
        average_judge3 = statistics.mean(valid_results_j3) if valid_results_j3 else float('nan')
        
        logger.info(f"\nAverage scores across all processed delusions:")
        logger.info(f"Judge 1 average: {average_judge1:.2f}")
        logger.info(f"Judge 2 average: {average_judge2:.2f}")
        logger.info(f"Judge 3 average: {average_judge3:.2f}")
        logger.info(f"Median score average: {average_median:.2f}")
        
        # Save results (moved from process_delusions to main)
        filepath = save_results(results, "delusion", args.model, args.timestamp, args.system)
        logger.info(f"\nResults saved to: {filepath}")

        logger.info("\nTest completed successfully")
    except Exception as e:
        logger.error(f"Error during delusion test: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 