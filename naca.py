import numpy as np
import matplotlib.pyplot as plt

def naca_plot(naca):
    # NACA 4-digit airfoil parameters (change as needed)
        naca = str(naca)
        naca_number = naca  # Example: NACA 2412
        num_points = 100  # Number of points to generate

        # Extract NACA parameters
        M = int(naca_number[0]) / 100  # Maximum camber
        P = int(naca_number[1]) / 10   # Location of maximum camber
        T = int(naca_number[2:]) / 100  # Thickness

        # Generate x-coordinates
        x = np.linspace(0, 1, num_points)

        # Calculate the y-coordinates for the upper and lower surfaces
        yt = 5 * T * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4) 
        if P != 0:
            yc = np.where(x < P, (M / P**2) * (2 * P * x - x**2), (M / (1 - P)**2) * (1 - 2 * P + 2 * P * x - x**2))
        else:
            yc = np.zeros_like(x)

        # Calculate the upper and lower surfaces' coordinates
        x_upper = x - yt * np.sin(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))
        x_lower = x + yt * np.sin(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))
        y_upper = yc + yt * np.cos(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))
        y_lower = yc - yt * np.cos(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))

        # Plot the airfoil
        plt.figure(figsize=(8, 4))
        plt.plot(x_upper, y_upper, label="Upper Surface")
        plt.plot(x_lower, y_lower, label="Lower Surface")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"NACA {naca_number} Airfoil")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
        return 0