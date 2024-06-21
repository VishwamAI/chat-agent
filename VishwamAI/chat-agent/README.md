# VishwamAI Chat Agent

## Overview
This project aims to develop the `VishwamAI` chat agent model with the goal of achieving 100% accuracy on the Massive Multitask Language Understanding (MMLU) benchmark, focusing on mathematical reasoning and other benchmarks. The project integrates advanced models such as LLaMA3 8b, 70b, Mixtral 8x7b, Gemma 7b, and the `grok-1` model to enhance performance and accuracy. TensorFlow T5 is the primary focus for tokenization and model architecture.

## Current Status
- Research and review of state-of-the-art methods, academic papers, and benchmarks for MMLU are ongoing.
- Environment setup and necessary packages for interacting with Papers with Code and extracting text from PDFs are complete.
- Key information from relevant model documentation, including the GPT-4 technical report and Gemini model papers, has been extracted and analyzed.
- Performance metrics graphs have been generated and stored as `/home/ubuntu/performance_metrics_graphs.png`.
- Detailed reports for individual MMLU-related papers are being compiled.
- Auto-update and continuous evaluation features have been implemented.
- Integration with the `grok-1` model is under assessment.
- TensorFlow T5 is the primary focus for tokenization and model architecture.
- Gradient checkpointing and mixed precision training have been introduced to address memory exhaustion issues.
- Mask tensor broadcasting logic has been corrected to ensure compatibility with `embedded_inputs`.

## Next Steps
- Finalize the integration with the `grok-1` model.
- Run the VishwamAI model training script using Python 3 to verify mask broadcasting resolution and identify new errors.
- Optimize memory usage during VishwamAI model training to prevent process termination.
- Test the VishwamAI model to ensure it achieves 100% MMLU and excels in MATH reasoning and other capabilities.

## Key Files
- `/home/ubuntu/chat-agent/VishwamAI/scripts/train_vishwamai_model.py`: Script to train the VishwamAI model.
- `/home/ubuntu/chat-agent/VishwamAI/scripts/config.py`: Configuration file for model settings.
- `/home/ubuntu/chat-agent/VishwamAI/scripts/model_architecture.py`: Model architecture definition for VishwamAI.
- `/home/ubuntu/chat-agent/VishwamAI/scripts/preprocess_data.py`: Script for data preprocessing.
- `/home/ubuntu/chat-agent/VishwamAI/scripts/train_sentencepiece_tokenizer.py`: Script to train the SentencePiece tokenizer.
- `/home/ubuntu/performance_metrics_graphs.png`: Performance metrics graphs for the VishwamAI model.

## Contact
For any questions or further information, please contact Devin AI and Kasinadhsarma.
