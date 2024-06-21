import matplotlib.pyplot as plt
import sys

def plot_memory_profile(file_path):
    timestamps = []
    memory_usage = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('MEM'):
                parts = line.split()
                timestamps.append(float(parts[1]))
                memory_usage.append(float(parts[2]))

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage, label='Memory Usage')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_memory_profile.py <path_to_dat_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    plot_memory_profile(file_path)
