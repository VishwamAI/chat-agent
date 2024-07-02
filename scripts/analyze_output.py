import re

def analyze_output(file_path, error_pattern, context_lines=5):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if re.search(error_pattern, line):
            start = max(i - context_lines, 0)
            end = min(i + context_lines + 1, len(lines))
            context = lines[start:end]
            print("".join(context))
            break

if __name__ == "__main__":
    error_pattern = r"TypeError: cannot reshape array.*"
    file_path = "/home/ubuntu/full_outputs/python3_train_py_1719938594.4505975.txt"
    analyze_output(file_path, error_pattern)
