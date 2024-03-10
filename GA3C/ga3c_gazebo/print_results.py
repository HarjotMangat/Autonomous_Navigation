import matplotlib.pyplot as plt

# Open the csv file and read the contents
with open('results.txt', 'r') as file:
    lines = file.readlines()

# Create empty lists for x-axis and y-axis values
x_values = []
y_values = []

# Loop through each line in the csv file
for i, line in enumerate(lines):
    # Split each line by comma
    values = line.strip().split(',')
    # Append the second value of each line to the y-axis list
    y_values.append(float(values[1]))
    # Append the line number to the x-axis list
    x_values.append(i+1)

# Plot the x-axis and y-axis values
plt.plot(x_values, y_values)

# Add title to the plot
plt.title('Episode Rewards')

# Add labels to the plot, x is the episode number, y is the reward
plt.xlabel('Episode')
plt.ylabel('Reward')

# Display the plot
plt.show()
