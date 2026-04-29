"""
This file demonstrates how to set up an evaluation pipeline using `ragas`.
We recommend running this via pytest locally or in a GitHub Actions pipeline.
"""
import pytest
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

# Example: Golden Dataset
# In a real scenario, this would be loaded from a JSON or CSV file specific to your domain.
golden_dataset_raw = {
    "question": [
        "What is the maximum allowed voltage for the sensor module?",
        "How do you calibrate the gyroscopic sensor?"
    ],
    "ground_truth": [
        "The maximum allowed voltage is 5V. [Doc 1]",
        "You calibrate the gyroscopic sensor by placing it on a flat surface and running the `calibrate_gyro()` function over I2C. [Doc 2]"
    ],
    # the exact correct context string that SHOULD be retrieved
    "contexts": [
        ["The sensor module operates at a standard 3.3V, but has a maximum allowed voltage tolerance of 5V."],
        ["To ensure accurate readings, place the device on a level surface. Send the calibration command by invoking `calibrate_gyro()` via the I2C interface."]
    ],
    # For evaluation, we would populate the pipeline's ACTUALLY generated answers and contexts
    "answer": [
        "The max voltage is 5V. [Doc 1]",
        "Place it flat and call calibrate_gyro(). [Doc 2]"
    ]
}

def test_ragas_metrics():
    """
    Evaluates our simulated pipeline outputs against the ground truth using Ragas metrics.
    """
    # Convert to HuggingFace dataset format required by ragas
    dataset = Dataset.from_dict(golden_dataset_raw)
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision
        ]
    )
    
    print("Ragas Evaluation Results:")
    print(result)
    
    # Example threshold assertions
    assert result["faithfulness"] > 0.85, "Pipeline is hallucinating too much!"
    assert result["answer_relevancy"] > 0.85, "Answers are not relevant enough to the queries!"

if __name__ == "__main__":
    test_ragas_metrics()
