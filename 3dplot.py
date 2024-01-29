import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the coordinates of the vertices of a cube
vertices = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
]

# Define the faces of the cube using vertex indices
faces = [
    [vertices[0], vertices[1], vertices[5], vertices[4]],
    [vertices[7], vertices[6], vertices[2], vertices[3]],
    [vertices[1], vertices[2], vertices[6], vertices[5]],
    [vertices[4], vertices[7], vertices[3], vertices[0]],
    [vertices[3], vertices[2], vertices[1], vertices[0]],
    [vertices[7], vertices[4], vertices[5], vertices[6]]
]

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a Poly3DCollection to plot the cube
cube = Poly3DCollection(faces, edgecolor='k', alpha=0.5)
ax.add_collection3d(cube)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)

# Show the 3D plot
plt.show()