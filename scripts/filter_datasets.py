def filter_datasets():
    keywords = ["math", "reasoning", "conversation", "dialogue", "Q&A"]
    relevant_datasets = []

    with open("/home/ubuntu/full_outputs/python3_list_dataset_1719478506.3049698.txt", "r") as file:
        for line in file:
            if any(keyword in line.lower() for keyword in keywords):
                relevant_datasets.append(line.strip())

    with open("/home/ubuntu/chat-agent/VishwamAI-main/scripts/relevant_datasets.txt", "w") as output_file:
        for dataset in relevant_datasets:
            output_file.write(dataset + "\n")

if __name__ == "__main__":
    filter_datasets()
