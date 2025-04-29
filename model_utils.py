import os
import requests
import re
import time
from typing import Tuple, Optional
import csv
from datetime import datetime
import statistics

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "google/gemini-2.0-flash-001"
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds

def extract_score(response: str) -> Optional[float]:
    """Extract the score from the response using the boxed notation."""
    match = re.search(r'\\boxed{(-?\d+(?:\.\d+)?)}', response)
    if match:
        return float(match.group(1))
    return None

def get_model_response(prompt: str, model: str = DEFAULT_MODEL) -> Tuple[str, Optional[float]]:
    """Get response from the model using OpenRouter API and extract score with retries."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return "", None
            
            content = response.json()["choices"][0]["message"]["content"]
            score = extract_score(content)
            
            if score is not None:
                return content, score
            
            print(f"Attempt {attempt + 1}: Could not extract score from response")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
    
    return "", None

def save_results(results: list, test_type: str, model: str):
    """Save results to a timestamped CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_type}_results_{model.replace('/', '_')}_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    return filename

def calculate_averages(results: list, score_fields: list) -> dict:
    """Calculate average scores for specified fields, excluding 'n/a' values."""
    averages = {}
    for field in score_fields:
        scores = [r[field] for r in results if r[field] != "n/a"]
        avg = statistics.mean(scores) if scores else float('nan')
        averages[field] = avg
    return averages 