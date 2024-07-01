import matplotlib.pyplot as plt

def parse_memory_profile(file_path):
    timestamps = []
    memory_usage = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("MEM"):
                parts = line.split()
                memory = float(parts[1])
                timestamp = float(parts[2])
                memory_usage.append(memory)
                timestamps.append(timestamp)

    return timestamps, memory_usage

def plot_memory_usage(timestamps, memory_usage):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage, label='Memory Usage (MB)')
    plt.xlabel('Timestamp')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/ubuntu/chat-agent/VishwamAI-main/memory_usage_plot.png')  # Save the plot as an image file

if __name__ == "__main__":
    file_path = "/home/ubuntu/chat-agent/VishwamAI-main/mprofile_20240628103001.dat"
    timestamps, memory_usage = parse_memory_profile(file_path)
    plot_memory_usage(timestamps, memory_usage)
