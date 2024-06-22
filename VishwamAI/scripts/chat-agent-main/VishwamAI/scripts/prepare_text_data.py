import csv

def extract_text_data(input_csv, output_txt):
    with open(input_csv, 'r') as csvfile, open(output_txt, 'w') as txtfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            txtfile.write(row['description_x'] + '\n')
            txtfile.write(row['description_y'] + '\n')

if __name__ == "__main__":
    input_csv = '/home/ubuntu/train.csv'
    output_txt = '/home/ubuntu/chat-agent/VishwamAI/scripts/text_data.txt'
    extract_text_data(input_csv, output_txt)
