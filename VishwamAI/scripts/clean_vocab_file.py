import re

def clean_vocab_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Remove non-printable characters and ensure unique key-value pairs
    cleaned_lines = []
    seen_keys = set()
    for line in lines:
        # Remove non-printable characters
        cleaned_line = re.sub(r'[^\x20-\x7E]+', '', line)
        if cleaned_line:
            key = cleaned_line.split()[0]
            if key not in seen_keys:
                cleaned_lines.append(cleaned_line + '\n')
                seen_keys.add(key)

    with open(output_file, 'w') as file:
        file.writelines(cleaned_lines)

if __name__ == "__main__":
    input_file = '../data/vishwamai.spm'
    output_file = '../data/vishwamai_cleaned.spm'
    clean_vocab_file(input_file, output_file)
