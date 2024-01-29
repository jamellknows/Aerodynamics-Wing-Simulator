import numpy as np
import matplotlib.pyplot as plt

def lift_drag_calc(rho=1.225, v_inf=10, chord_length=1.0, alpha_start=-10, alpha_increment=20, alpha_end=100):
    
    # modify to use naca 
    # Constants
    rho = rho  # Air density (kg/m^3)
    v_inf = v_inf  # Freestream velocity (m/s)
    chord_length = chord_length  # Airfoil chord length (m)
    alpha_range = np.linspace(alpha_start, alpha_increment, alpha_end)  # Angle of attack range (degrees)

    # Airfoil Geometry
    def airfoil_thickness(x):
        # Simplified airfoil thickness distribution (modify as needed)
        return 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    # Calculate lift and drag coefficients at various angles of attack
    lift_coefficients = []
    drag_coefficients = []

    for alpha in alpha_range:
        alpha_rad = np.radians(alpha)
        
        # Compute airfoil coordinates
        x_coordinates = np.linspace(0, 1, 100)  # 100 points along the chord
        y_coordinates = airfoil_thickness(x_coordinates) * np.sin(alpha_rad)
        
        # Calculate lift and drag
        lift =  np.trapz(y_coordinates, x_coordinates) * rho * v_inf**2
        drag = 2 * np.trapz(y_coordinates, x_coordinates) * rho * v_inf**2 * np.cos(alpha_rad)
        
        lift_coefficient = lift / (0.5 * rho * v_inf**2 * chord_length)
        drag_coefficient = drag / (0.5 * rho * v_inf**2 * chord_length)
        
        lift_coefficients.append(lift_coefficient)
        drag_coefficients.append(drag_coefficient)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_range, lift_coefficients, label='Lift Coefficient')
    plt.plot(alpha_range, drag_coefficients, label='Drag Coefficient')
    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Coefficient Value')
    plt.title('Airfoil Lift and Drag Coefficients')
    plt.grid(True)
    plt.legend()
    plt.show()
    return 
