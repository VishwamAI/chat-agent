# Chat Agent Prompts Guide

This document provides examples of interactive prompts that the chat agent can use to engage with users. These prompts are designed to make the chat agent more interactive and user-friendly.

## Greeting Prompts

When a user initiates a conversation with a greeting, the chat agent can respond with a friendly greeting and an offer to assist. Here are some examples:

**User:** Hi
**Chat Agent:** Hello! How can I assist you today?

**User:** Hello
**Chat Agent:** Hi there! What can I help you with today?

**User:** Hey
**Chat Agent:** Hey! How can I help you today?

## Assistance Prompts

When a user asks for help or has a question, the chat agent can respond with prompts that guide the user to provide more information or offer solutions. Here are some examples:

**User:** I need help with my account
**Chat Agent:** Sure, I can help with that. Can you please provide more details about the issue you're facing with your account?

**User:** How do I reset my password?
**Chat Agent:** To reset your password, please follow these steps: [provide steps]. If you need further assistance, let me know!

**User:** Can you help me with my order?
**Chat Agent:** Of course! Please provide your order number, and I'll look into it for you.

## Informational Prompts

When a user requests information, the chat agent can provide the requested information or direct the user to relevant resources. Here are some examples:

**User:** What are your business hours?
**Chat Agent:** Our business hours are Monday to Friday, 9 AM to 5 PM. If you have any other questions, feel free to ask!

**User:** Where are you located?
**Chat Agent:** We are located at [address]. You can find more details on our website [link].

**User:** How can I contact customer support?
**Chat Agent:** You can contact our customer support team via email at [email address] or call us at [phone number]. We're here to help!

## Closing Prompts

When a conversation is coming to an end, the chat agent can use closing prompts to ensure the user feels satisfied with the interaction. Here are some examples:

**User:** That's all I needed, thanks!
**Chat Agent:** You're welcome! If you have any other questions in the future, feel free to reach out. Have a great day!

**User:** Thanks for your help
**Chat Agent:** My pleasure! If you need any further assistance, don't hesitate to ask. Goodbye!

**User:** Bye
**Chat Agent:** Goodbye! Have a wonderful day!

## Custom Scenarios

If you have specific scenarios or use cases in mind that you want the chat agent to handle, please share them, and I'll include those in the documentation as well.

---

This guide provides a foundation for creating interactive and engaging prompts for the chat agent. Feel free to customize these prompts to better suit your specific needs and use cases.

## Running the Model with a Smaller Dataset

To run the VishwamAI chat agent model with a smaller dataset, follow these steps:

### Step 1: Prepare the Dataset

1. **CSV Files**: Ensure your dataset is in CSV format. Each row should represent a training example, with columns for input text and target text.
2. **Data from the Internet**: If sourcing data from the internet, preprocess and format it into a CSV file.

### Step 2: Set Up the Environment

1. **Install Dependencies**: Ensure all dependencies are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

### Step 3: Configure the Training Script

1. **Configuration File**: Ensure the `default_config.yaml` file is correctly set up with the necessary parameters, including the path to your dataset.

### Step 4: Run the Training Script

1. **Execute the Script**: Run the `train.py` script located in the `scripts` directory:
   ```bash
   python3 scripts/train.py
   ```

### Step 5: Monitor the Training Process

1. **Training Output**: Monitor the training process through the console output. Ensure the training loss is decreasing, indicating the model is learning correctly.

### Step 6: Evaluate the Model

1. **Evaluation**: After training, evaluate the model's performance using the `evaluate.py` script to ensure it meets the specified requirements.

By following these steps, you can successfully run the VishwamAI chat agent model with a smaller dataset and monitor its training process.
