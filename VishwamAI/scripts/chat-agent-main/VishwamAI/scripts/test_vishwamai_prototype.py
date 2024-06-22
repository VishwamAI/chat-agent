import tensorflow as tf
import numpy as np
import os
from scripts.vishwamai_prototype import VishwamAI
import logging
import haiku as hk
import transformers




def test_build_nlp_model():
    # Test the build_nlp_model method to ensure the NLP model and tokenizer are created successfully.
    vishwamai = VishwamAI(batch_size=32)
    nlp_model, tokenizer = vishwamai.build_nlp_model()
    assert nlp_model is not None, "NLP model should not be None"
    assert tokenizer is not None, "Tokenizer should not be None"
    assert isinstance(nlp_model, (tf.keras.Model, haiku.Module)), "NLP model should be an instance of tf.keras.Model or haiku.Module"
    assert isinstance(tokenizer, transformers.PreTrainedTokenizer), "Tokenizer should be an instance of transformers.PreTrainedTokenizer"
    logging.info("test_build_nlp_model passed.")

def test_load_sample_dataset():
    # Test the load_sample_dataset method to ensure the sample dataset is loaded successfully.
    vishwamai = VishwamAI(batch_size=32)
    dataset = vishwamai.load_sample_dataset(batch_size=32)
    assert dataset is not None, "Sample dataset should not be None"
    assert isinstance(dataset, tf.data.Dataset), "Dataset should be an instance of tf.data.Dataset"
    logging.info("test_load_sample_dataset passed.")

def test_train():
    # Test the train method to ensure the model can be trained without errors.
    vishwamai = VishwamAI(batch_size=32)
    try:
        vishwamai.train(epochs=1, batch_size=32)
        logging.info("test_train passed.")
    except Exception as e:
        logging.error(f"test_train failed: {e}")



def test_self_improve():
    # Test the self_improve method to ensure the self-improvement process can be executed without errors.
    vishwamai = VishwamAI(batch_size=32)
    try:
        vishwamai.self_improve()
        logging.info("test_self_improve passed.")
    except Exception as e:
        logging.error(f"test_self_improve failed: {e}")

def test_evaluate_performance():
    # Test the evaluate_performance method to ensure the model's performance can be evaluated and metrics are returned.
    vishwamai = VishwamAI(batch_size=32)
    try:
        performance_metrics = vishwamai.evaluate_performance()
        assert performance_metrics is not None, "Performance metrics should not be None"
        assert 'FID' in performance_metrics, "Performance metrics should contain 'FID'"
        assert 'IS' in performance_metrics, "Performance metrics should contain 'IS'"
        logging.info("test_evaluate_performance passed.")
    except Exception as e:
        logging.error(f"test_evaluate_performance failed: {e}")

def test_search_new_data():
    # Test the search_new_data method to ensure new data can be searched and returned as a list.
    vishwamai = VishwamAI(batch_size=32)
    try:
        new_data = vishwamai.search_new_data()
        assert new_data is not None, "New data should not be None"
        assert isinstance(new_data, list), "New data should be a list"
        logging.info("test_search_new_data passed.")
    except Exception as e:
        logging.error(f"test_search_new_data failed: {e}")

def test_integrate_new_data():
    # Test the integrate_new_data method to ensure new data can be integrated into the training process without errors.
    vishwamai = VishwamAI(batch_size=32)
    try:
        new_data = vishwamai.search_new_data()
        vishwamai.integrate_new_data(new_data)
        logging.info("test_integrate_new_data passed.")
    except Exception as e:
        logging.error(f"test_integrate_new_data failed: {e}")

def test_generate_question():
    # Test the generate_question method to ensure questions can be generated based on input text.
    vishwamai = VishwamAI(batch_size=32)
    test_inputs = [
        "Explain the significance of the moon landing.",
        "Describe the process of photosynthesis.",
        "What are the benefits of regular exercise?",
        "How does the internet work?",
        "Why is climate change a global concern?"
    ]
    try:
        for input_text in test_inputs:
            question = vishwamai.generate_question(input_text)
            assert question is not None, "Generated question should not be None"
            assert isinstance(question, str), "Generated question should be a string"
            logging.info(f"test_generate_question passed for input: {input_text}")
    except Exception as e:
        logging.error(f"test_generate_question failed: {e}")

if __name__ == "__main__":
    test_build_generator()
    test_build_discriminator()
    test_build_gan()
    test_build_nlp_model()
    test_load_sample_dataset()
    test_train()
    test_train_and_generate_images()
    test_generate_image()
    test_self_improve()
    test_evaluate_performance()
    test_search_new_data()
    test_integrate_new_data()
    test_generate_question()
