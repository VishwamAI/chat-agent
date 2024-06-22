import os
import subprocess
import logging
from datetime import datetime
from model_architecture import VishwamAIModel, ScoringSystem

# Set up logging
logging.basicConfig(filename='/home/ubuntu/home/ubuntu/VishwamAI/logs/performance.log', level=logging.INFO)

def run_evaluation():
    model = VishwamAIModel()
    scoring_system = ScoringSystem()

    # Example evaluation input
    example_input = "What is the capital of France?"
    output = model(example_input)

    # Example scoring update
    correct = True  # This would be determined by the model's output in a real scenario
    question_type = "medium"  # Example question type
    new_score = scoring_system.update_score(correct, question_type)

    # Log the results
    logging.info(f"{datetime.now()} - Input: {example_input}, Output: {output}, Updated score: {new_score}")

    # Perform self-improvement
    model.self_improve()

    # Log self-improvement results
    logging.info(f"{datetime.now()} - Self-improvement performed")

if __name__ == "__main__":
    run_evaluation()
