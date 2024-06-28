import random

def generate_modular_question(modules):
    """
    Generate a modular question by chaining modules with matching input and output types.

    Args:
        modules (list): A list of modules, each represented as a dictionary with 'input_type' and 'output_type'.

    Returns:
        str: A generated question composed of chained modules.
    """
    question = ""
    current_output_type = None

    for module in modules:
        if current_output_type is None or module['input_type'] == current_output_type:
            question += module['question'] + " "
            current_output_type = module['output_type']
        else:
            raise ValueError("Module input type does not match the previous module's output type.")

    return question.strip()

if __name__ == "__main__":
    # Example modules for testing
    example_modules = [
        {'input_type': None, 'output_type': 'number', 'question': "What is 2 + 2?"},
        {'input_type': 'number', 'output_type': 'number', 'question': "Multiply the result by 3."},
        {'input_type': 'number', 'output_type': 'number', 'question': "Subtract 5 from the result."}
    ]

    try:
        generated_question = generate_modular_question(example_modules)
        print("Generated Question:", generated_question)
    except ValueError as e:
        print("Error:", e)
