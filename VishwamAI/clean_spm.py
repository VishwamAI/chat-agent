with open('data/vishwamai.spm', 'r') as file:
    lines = file.readlines()
    lines = [line for line in lines if line.strip()]

with open('data/vishwamai.spm', 'w') as file:
    file.writelines(lines)
