import re

def analyze_output(file_path, error_pattern, context_lines=5):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    print(f"Total lines read: {len(lines)}")
    print("Last few lines of the file:")
    for line in lines[-10:]:
        print(line.strip())

    for i, line in enumerate(lines):
        if re.search(error_pattern, line):
            start = max(i - context_lines, 0)
            end = min(i + context_lines + 1, len(lines))
            context = lines[start:end]
            print("".join(context))

if __name__ == "__main__":
    error_pattern = r"TypeError: cannot reshape array.*"
    file_path = "/home/ubuntu/chat-agent/logs/train.log"
    analyze_output(file_path, error_pattern)
