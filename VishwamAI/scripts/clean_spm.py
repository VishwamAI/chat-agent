import re

def clean_spm_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove non-printable characters
        cleaned_line = re.sub(r'[^\x20-\x7E]+', '', line)
        cleaned_lines.append(cleaned_line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

if __name__ == "__main__":
    input_file = "/home/ubuntu/chat-agent/VishwamAI/data/vishwamai.spm"
    output_file = "/home/ubuntu/chat-agent/VishwamAI/data/vishwamai_cleaned.spm"
    clean_spm_file(input_file, output_file)
    print(f"Cleaned file saved to {output_file}")
