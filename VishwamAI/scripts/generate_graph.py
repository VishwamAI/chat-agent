import matplotlib.pyplot as plt
import numpy as np

# Sample data for the graph
time = np.linspace(0, 40, 100)
memory_usage = np.linspace(0, 4500, 100)

# Create the figure and axes
fig, ax = plt.subplots()
ax.plot(time, memory_usage, color='white')

# Set the background color and grid lines
ax.set_facecolor('black')
ax.grid(color='white', linestyle='--', linewidth=0.5)

# Set the labels and title
ax.set_xlabel('Time (in seconds)', color='white')
ax.set_ylabel('Memory (in MB)', color='white')
ax.set_title('python3 train_vishwamai_model.py', color='white')

# Set the x and y axis limits
ax.set_xlim([0, 40])
ax.set_ylim([0, 4500])

# Add the red dashed lines
ax.axvline(x=20, color='red', linestyle='--')
ax.axhline(y=4000, color='red', linestyle='--')

# Add the text annotation
ax.text(20, 4200, '20 GB / 32024 start at 21:20:36.746', color='white', fontsize=8)

# Set the tick parameters
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Save the graph as an image file
plt.savefig('/home/ubuntu/chat-agent/VishwamAI/graphs/memory_usage_graph.png', bbox_inches='tight', facecolor='black')

# Show the plot
plt.show()
