import tensorflow as tf
import torch
import pickle
from sklearn.metrics import accuracy_score
from model_architecture import VishwamAIModel, ScoringSystem

def load_preprocessed_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def evaluate_model(model, data):
    predictions = []
    labels = []
    scoring_system = ScoringSystem()
    for inputs, label, question_type in data:
        output = model(inputs)
        prediction = tf.round(output).numpy()
        predictions.append(prediction)
        labels.append(label)
        correct = prediction == label
        scoring_system.update_score(correct, question_type)
    accuracy = accuracy_score(labels, predictions)
    final_score = scoring_system.score
    return accuracy, final_score

if __name__ == "__main__":
    # Load the preprocessed data
    train_data = load_preprocessed_data("../data/train_data.pkl")
    val_data = load_preprocessed_data("../data/val_data.pkl")
    test_data = load_preprocessed_data("../data/test_data.pkl")

    # Initialize the model
    model = VishwamAIModel()

    # Evaluate the model on the validation and test sets
    val_accuracy, val_score = evaluate_model(model, val_data)
    test_accuracy, test_score = evaluate_model(model, test_data)

    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Score: {val_score}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Score: {test_score}")
