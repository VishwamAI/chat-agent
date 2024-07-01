import matplotlib.pyplot as plt

def parse_large_memory_usage_file(file_path, chunk_size=1024):
    memory_usage = []
    with open(file_path, 'r') as f:
        while True:
            lines = f.readlines(chunk_size)
            if not lines:
                break
            for line in lines:
                if 'MiB Mem' in line:
                    parts = line.split()
                    if len(parts) > 9:
                        # Extract the used memory value in MiB
                        memory_usage.append(float(parts[5]))
    return memory_usage

def plot_memory_usage(memory_usage, output_file):
    # Generate a time series for the x-axis
    time_series = list(range(len(memory_usage)))
    plt.plot(time_series, memory_usage)
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MiB)')
    plt.title('Memory Usage During Training')
    plt.savefig(output_file)

if __name__ == "__main__":
    memory_usage_file = '/home/ubuntu/memory_usage.txt'
    output_image_file = '/home/ubuntu/memory_usage_plot.png'
    memory_usage_data = parse_large_memory_usage_file(memory_usage_file)
    plot_memory_usage(memory_usage_data, output_image_file)
