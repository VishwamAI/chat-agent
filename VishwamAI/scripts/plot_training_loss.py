import matplotlib.pyplot as plt
import re

# Function to parse the training log file and extract epoch numbers and loss values
def parse_training_log(log_file):
    epochs = []
    losses = []
    with open(log_file, 'r') as file:
        for line in file:
            match = re.search(r'Epoch (\d+), Loss: ([\d\.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)
    return epochs, losses

# Function to plot the training loss over epochs
def plot_training_loss(log_file):
    epochs, losses = parse_training_log(log_file)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    plt.show()

if __name__ == "__main__":
    log_file = '../logs/training_run_log.txt'
    plot_training_loss(log_file)
