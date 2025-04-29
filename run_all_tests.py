import argparse
import subprocess
import csv
from datetime import datetime
import os
from model_utils import calculate_averages
import statistics
import math
import logging
from typing import TextIO

def setup_logging(model: str) -> tuple[logging.Logger, str]:
    """Set up logging to both file and console."""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create logger
    logger = logging.getLogger('sycophancy_tests')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/test_run_{model.replace('/', '_')}_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def run_test(script_name: str, model: str, limit: int, logger: logging.Logger) -> dict:
    """Run a test script and return its results."""
    logger.info(f"\nRunning {script_name}...")
    cmd = ["python", script_name, "--model", model]
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Error running {script_name}:")
        logger.error(result.stderr)
        return None
    
    # Log the full output
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    
    # Parse the output to get the results file name
    output_lines = result.stdout.split('\n')
    results_file = None
    for line in output_lines:
        if line.startswith("Results saved to:"):
            results_file = line.split(": ")[1].strip()
            break
    
    if not results_file or not os.path.exists(results_file):
        logger.error(f"Could not find results file for {script_name}")
        return None
    
    return {"results_file": results_file}

def process_pickside_results(filename: str, logger: logging.Logger) -> float:
    """Process pickside results to get average score."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    scores1 = [float(r['score1']) for r in results if r['score1'] != 'n/a']
    scores2 = [float(r['score2']) for r in results if r['score2'] != 'n/a']
    avg1 = statistics.mean(scores1) if scores1 else float('nan')
    avg2 = statistics.mean(scores2) if scores2 else float('nan')
    overall_avg = (avg1 + avg2) / 2 if not math.isnan(avg1) and not math.isnan(avg2) else float('nan')
    logger.info(f"Processed {len(scores1)} valid scores from pickside test")
    return overall_avg

def process_mirror_results(filename: str, logger: logging.Logger) -> float:
    """Process mirror results to get average difference in scores."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    differences = []
    for r in results:
        if r['score1'] != 'n/a' and r['score2'] != 'n/a':
            diff = abs(float(r['score1']) - float(r['score2']))
            differences.append(diff)
    
    avg = statistics.mean(differences) if differences else float('nan')
    logger.info(f"Processed {len(differences)} valid score pairs from mirror test")
    return avg

def process_iq_results(filename: str, logger: logging.Logger) -> dict:
    """Process IQ results to get all statistics."""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    # Get valid IQ scores
    valid_results = [r for r in results if r['iq_estimate'] != 'n/a']
    iq_scores = [float(r['iq_estimate']) for r in valid_results]
    
    if not iq_scores:
        logger.warning("No valid IQ scores found in results")
        return {
            'overall_average': float('nan'),
            'very_low_average': float('nan'),
            'low_average': float('nan'),
            'average_average': float('nan'),
            'high_average': float('nan'),
            'refusal_rate': 1.0
        }
    
    # Calculate category averages based on predefined categories
    category_scores = {
        'very_low': [],
        'low': [],
        'average': [],
        'high': []
    }
    
    for result in valid_results:
        category = result['category']
        if category in category_scores:
            category_scores[category].append(float(result['iq_estimate']))
    
    # Calculate averages for each category
    category_averages = {
        f'{cat}_average': statistics.mean(scores) if scores else float('nan')
        for cat, scores in category_scores.items()
    }
    
    # Calculate overall average of category averages
    valid_category_averages = [avg for avg in category_averages.values() if not math.isnan(avg)]
    overall_average = statistics.mean(valid_category_averages) if valid_category_averages else float('nan')
    
    # Calculate refusal rate
    refusal_rate = 1 - (len(valid_results) / len(results))
    
    # Log category statistics
    logger.info(f"Processed {len(valid_results)} valid IQ scores out of {len(results)} total responses")
    logger.info(f"Refusal rate: {refusal_rate:.1%}")
    for cat, scores in category_scores.items():
        logger.info(f"{cat.replace('_', ' ').title()} IQ range: {len(scores)} responses")
    
    return {
        'overall_average': overall_average,
        **category_averages,
        'refusal_rate': refusal_rate
    }

def main():
    parser = argparse.ArgumentParser(description='Run all sycophancy tests')
    parser.add_argument('--model', type=str, default='google/gemini-2.0-flash-001',
                      help='Model to use for testing')
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit the number of samples to process in each test')
    args = parser.parse_args()
    
    # Set up logging
    logger, log_filename = setup_logging(args.model)
    logger.info(f"Starting test run for model: {args.model}")
    logger.info(f"Log file: {log_filename}")
    if args.limit:
        logger.info(f"Limiting each test to {args.limit} samples")
    
    # Run all tests
    tests = {
        'pickside': 'test_pickside.py',
        'mirror': 'test_mirror.py',
        'iq': 'test_iq.py'
    }
    
    results = {}
    for test_name, script_name in tests.items():
        test_result = run_test(script_name, args.model, args.limit, logger)
        if test_result:
            results[test_name] = test_result
    
    # Process results
    master_results = {
        'model': args.model,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add individual test results
    for test_name in tests:
        if test_name in results:
            results_file = results[test_name]['results_file']
            master_results[f'{test_name}_results_file'] = results_file
            
            if test_name == 'pickside':
                master_results['pickside_average'] = process_pickside_results(results_file, logger)
            elif test_name == 'mirror':
                master_results['mirror_difference'] = process_mirror_results(results_file, logger)
            elif test_name == 'iq':
                iq_stats = process_iq_results(results_file, logger)
                master_results.update(iq_stats)
    
    # Save master results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_filename = f"master_results_{args.model.replace('/', '_')}_{timestamp}.csv"
    
    with open(master_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=master_results.keys())
        writer.writeheader()
        writer.writerow(master_results)
    
    logger.info(f"\nMaster results saved to: {master_filename}")
    
    # Print summary
    logger.info("\nSummary of Results:")
    logger.info(f"Model: {args.model}")
    for test_name in tests:
        if test_name in results:
            if test_name == 'pickside':
                logger.info(f"Pickside Average Score: {master_results['pickside_average']:.2f}")
            elif test_name == 'mirror':
                logger.info(f"Mirror Average Difference: {master_results['mirror_difference']:.2f}")
            elif test_name == 'iq':
                logger.info("\nIQ Test Results:")
                logger.info(f"Overall Average: {master_results['overall_average']:.2f}")
                logger.info(f"Very Low IQ Average: {master_results['very_low_average']:.2f}")
                logger.info(f"Low IQ Average: {master_results['low_average']:.2f}")
                logger.info(f"Average IQ Average: {master_results['average_average']:.2f}")
                logger.info(f"High IQ Average: {master_results['high_average']:.2f}")
                logger.info(f"Refusal Rate: {master_results['refusal_rate']:.1%}")
    
    logger.info("\nTest run completed successfully")

if __name__ == "__main__":
    main() 