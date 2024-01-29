import numpy as np
import matplotlib.pyplot as plt
# run as sqplot.square()
def square(l, x_value, y_value):
    # Create a grid of x and y values
    x_min = x_value - l/2
    x_max = x_value + l/2 
    y_min = y_value - l/2 
    y_max = y_value + l/2 
    x_values = np.linspace(x_min, x_max, 100)
    y_values = np.linspace(y_min, y_max, 100)
    
    x, y = np.meshgrid(x_values, y_values)
    
    # Calculate the function values for each (x, y) pair
    z = np.sqrt(l - (l/2 - x) - (l/2 - y))
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    ax.plot_surface(x, y, z, cmap='viridis')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('sqrt(l - (l/2 - x) - (l/2 - y))')
    
    # Show the plot
    plt.show()
    return True



