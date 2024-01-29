import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def windsim(v_inf, naca, alpha, time, chord_length, thickness_range, plot=False):
    #initialise aerofoil
    naca_number = str(naca)
    M = int(naca_number[0]) / 100  # Maximum camber
    P = int(naca_number[1]) / 10   # Location of maximum camber
    T = int(naca_number[2:]) / 100
    num_points = 100
    chord_length = float(chord_length)
    x = np.linspace(1, 1+chord_length, num_points)
    alpha = np.radians(alpha)
    time_array = [i for i in range(1, time+1)]
    T = int(T * thickness_range)
    # initialise arrays
    dynamic_pressure = []
    total_pressure = []
    dynamic_viscocity = []
    kinematic_viscocity = []
    velocity_gradient = [v_inf]
    velocity_coefficient = []
    pressure_gradient = []
    pressure_coefficients = []
    boundary_velocity = []
    lift_gradients = []
    lift_coefficients =[]
    drag_gradients = []
    drag_coefficients = []
    
    # constants
    rho = 1.225
    g_acc = 9.81
    conventional_dynamic_viscocity = 1.789*(10**(-5))
    static_pressure = rho * g_acc * T
    
    def y_calculations(alpha):
        yt = 5 * T * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4) 
        if P != 0:
            yc = np.where(x < P, (M / P**2) * (2 * P * x - x**2), (M / (1 - P)**2) * (1 - 2 * P + 2 * P * x - x**2))
        else:                
            yc = np.zeros_like(x)
        x_upper = x - yt * np.sin(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))*np.cos(alpha)
        x_lower = x + yt * np.sin(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))*np.cos(alpha)
        y_upper = yc + yt * np.cos(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))*np.sin(alpha)
        y_lower = yc - yt * np.cos(np.arctan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * x)))*np.sin(alpha)

        return x_upper, x_lower, y_upper, y_lower
    

    
    def create_nested_array(rows, columns, depth, default_value=None):
        nested_array = [[[0 for _ in range(rows)] for _ in range(columns)] for _ in range(depth)]
        return nested_array 

    def Dynamic_pressure(v):
        d_p = 0.5 * rho * velocity_gradient[-1]**2 
        dynamic_pressure.append(d_p)        
        return d_p
        
    def Total_pressure(static_pressure, dynamic_pressure):
        t_p = static_pressure + dynamic_pressure
        total_pressure.append(t_p)
        return t_p
    
    def Dynamic_viscocity(dynamic_pressure, x_i, v_i):
       d_v = dynamic_pressure * (x_i/v_i)
       dynamic_viscocity.append(d_v)
       return d_v
       
    def Kinematic_viscosity(dynamic_viscosity, velocity, rho):
        k_v = dynamic_viscosity/rho
        kinematic_viscocity.append(k_v)
        return  k_v
    
    def Boundary_velocity(kinematic_viscocity, x_i, t_i):
        if t_i != 0 and x_i != 0:
            b_d = ((x_i/2)*t_i) + (x_i/2) - (kinematic_viscocity/x_i)
            boundary_velocity.append(b_d)
        elif t_i == 0:
            b_d = ((x_i/2)*(t_i+1)) + (x_i/2) - (kinematic_viscocity/x_i)
            boundary_velocity.append(b_d)
        elif x_i == 0:
            b_d = ((x_i/2)*t_i) + (x_i/2) - (kinematic_viscocity/(x_i+1))
            boundary_velocity.append(b_d)
        if math.isnan(b_d):
            b_d = 1
            b_d_c = b_d/v_inf
        b_d_c = b_d/v_inf

            
        return b_d, b_d_c
    
    def Velocity_gradient(boundary_velocity, total_pressure):
        v_i = 0
        if boundary_velocity == 0:
            v_i = total_pressure/1
        else:
            v_i = total_pressure/boundary_velocity
        velocity_gradient.append(v_i)
        velocity_coefficient = v_i/v_inf
        return v_i, velocity_coefficient
    
    def Pressure_gradient(boundary_viscocity, velocity_gradient):
        p_i = -velocity_gradient*boundary_viscocity
        pressure_gradient.append(p_i)
        pc_i = p_i/(0.5 * rho * velocity_gradient**2)
        return p_i, pc_i
    
    def Lift_gradient(velocity_gradient, x_ii):
        lift =  np.trapz(velocity_gradient, x_ii) * rho
        lift_coefficient = lift / (0.5 * rho * v_inf**2)
        return lift, lift_coefficient

    def Drag_gradient(velocity_gradient, x_i):
        drag = 2 * np.trapz(velocity_gradient, x_i) * rho
        drag_coefficient = drag / (0.5 * rho * v_inf**2)
        return drag, drag_coefficient
        

    def Flow_in_xt(velocity_gradient, alpha, x):
        x_upper, x_lower, y_upper, y_lower = y_calculations(alpha)
        surfaces = [x_upper, x_lower, y_upper, y_lower]
        rows = len(time_array)
        cols = len(x)
        depth = len(surfaces)
        lift_array = create_nested_array(rows, cols, depth)
        drag_array = create_nested_array(rows, cols, depth)
        velocity_array = create_nested_array(rows, cols, depth)
        pressure_array = create_nested_array(rows, cols, depth)
        boundary_array = create_nested_array(rows, cols, depth)
        lift_coefficient_array = create_nested_array(rows, cols, depth)
        drag_coefficient_array = create_nested_array(rows, cols, depth)
        velocity_coefficient_array = create_nested_array(rows, cols, depth)
        pressure_coefficient_array = create_nested_array(rows, cols, depth)
        boundary_coefficient_array = create_nested_array(rows, cols, depth)
        position_array = [i for i in range(1, 101)]
        row = -1
        
        

        for t in time_array:
           
            dep = 0
            d_p = Dynamic_pressure(velocity_gradient[t-1])
            t_p = Total_pressure(static_pressure, d_p)
            for surface in surfaces: 
                col = 0
                row = row + 1
                x = surface
                for x_i in x:

                    d_v = Dynamic_viscocity(d_p, x_i, velocity_gradient[t-1])
                    k_v = Kinematic_viscosity(d_v, rho, velocity_gradient[t-1])
                    b_v, b_v_c = Boundary_velocity(k_v, x_i, t)
                    v_i, vc_i = Velocity_gradient(b_v, t_p)
                    p_i, pc_i = Pressure_gradient(b_v, v_i)
                    x_ii = np.linspace(1, x_i, len(velocity_gradient))
                    l_i, lc_i = Lift_gradient(velocity_gradient, x_ii)
                    d_i, dc_i = Drag_gradient(velocity_gradient, x_ii)
                    lift_array[row][col][dep] = l_i
                    drag_array[row][col][dep] = d_i
                    velocity_array[row][col][dep] = v_i
                    pressure_array[row][col][dep] = p_i 
                    boundary_array[row][col][dep] = b_v
                    lift_coefficient_array[row][col][dep] = lc_i
                    drag_coefficient_array[row][col][dep] = dc_i
                    velocity_coefficient_array[row][col][dep] = vc_i
                    pressure_coefficient_array[row][col][dep] = pc_i
                    boundary_coefficient_array[row][col][dep] = b_v_c
                    col = col + 1
            dep = dep + 1
            return lift_array, drag_array, velocity_array, pressure_array, boundary_array, position_array, lift_coefficient_array, drag_coefficient_array, velocity_coefficient_array, pressure_coefficient_array, boundary_coefficient_array
        
    def Resultant_flows(lift_array, drag_array, velocity_array, pressure_array, boundary_array):
        x_upper, x_lower, y_upper, y_lower = y_calculations(alpha)
        x_resultant = (x_upper + x_lower)/2
        y_resultant = (y_upper + y_lower) /2
        resultant = np.sqrt(np.square(x_resultant) + np.square(y_resultant**2))
        resultant_surfaces = [x_resultant, y_resultant, resultant]
        la_xr = (np.array(lift_array[0]) + np.array(lift_array[1]))/2
        da_xr = (np.array(drag_array[0]) + np.array(drag_array[1]))/2
        va_xr = (np.array(velocity_array[0]) + np.array(velocity_array[1]))/2
        pa_xr = (np.array(pressure_array[0]) + np.array(pressure_array[1]))/2
        ba_xr = (np.array(boundary_array[0]) + np.array(boundary_array[1]))/2
        la_yr = (np.array(lift_array[2]) + np.array(lift_array[3]))/2
        da_yr = (np.array(lift_array[2]) + np.array(lift_array[3]))/2
        va_yr = (np.array(lift_array[2]) + np.array(lift_array[3]))/2
        pa_yr = (np.array(lift_array[2]) + np.array(lift_array[3]))/2
        ba_yr = (np.array(lift_array[2]) + np.array(lift_array[3]))/2
        la_r = np.sqrt(np.square((la_xr) + np.square(la_yr)))
        da_r = np.sqrt(np.square((da_xr) + np.square(da_yr)))
        va_r = np.sqrt(np.square((va_xr) + np.square(va_yr)))
        pa_r = np.sqrt(np.square((pa_xr) + np.square(pa_yr)))
        ba_r = np.sqrt(np.square((ba_xr) + np.square(ba_yr)))
        # calculate coefficients
    
        
        lift_array = np.concatenate((la_xr[np.newaxis, ...], la_yr[np.newaxis, ...], la_r[np.newaxis, ...]), axis=0)
        drag_array = np.concatenate((da_xr[np.newaxis, ...], da_yr[np.newaxis, ...], da_r[np.newaxis, ...]), axis=0)
        velocity_array = np.concatenate((va_xr[np.newaxis, ...], va_yr[np.newaxis, ...], va_r[np.newaxis, ...]), axis=0)
        pressure_array = np.concatenate((pa_xr[np.newaxis, ...], pa_yr[np.newaxis, ...], pa_r[np.newaxis, ...]), axis=0)
        boundary_array = np.concatenate((ba_xr[np.newaxis, ...], ba_yr[np.newaxis, ...], ba_r[np.newaxis, ...]), axis=0)
        # 0 is x_lower 1 is x _upper 2 is y _lower 3 is y upper
        # calcuate arrays
        return lift_array, drag_array, velocity_array, pressure_array, boundary_array
    
    def Plot(time_array, position_array, lift_array, drag_array, velocity_array, pressure_array, boundary_array, alpha):
        time_array = np.array(time_array)
        position_array = np.array(position_array)
     
        X, Y = np.meshgrid(position_array, time_array)
     
        surface_names = ['x_upper', 'x_lower', 'y_upper', 'y_lower', 'x_resultant', 'y_resultant', 'airfoil_total']
        lift_array, drag_array, velocity_array, pressure_array, boundary_array = Resultant_flows(lift_array, drag_array, velocity_array, pressure_array, boundary_array)
        surface_index = 0
        for i in range(0,len(surface_names)):
            
            
            #depth, position, time
            surface_lift_array = np.array(lift_array[i])
            surface_drag_array = np.array(drag_array[i])
            surface_velocity_array = np.array(velocity_array[i])
            surface_pressure_array = np.array(pressure_array[i])
            surface_boundary_array = np.array(boundary_array[i])
            z_data = [surface_lift_array, surface_drag_array, surface_velocity_array, surface_pressure_array, surface_boundary_array]
            index = 0
            for z_i in z_data:
                for j in range(0,len(z_data)):
                    Z = np.transpose(z_i)
                    print(z_i.shape)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(X, Y, Z, cmap='viridis', antialiased=True)
                    ax.set_xlabel('Chord position in percent')
                    ax.set_ylabel('Time in unit seconds')
                    z_label_name_set = ['Lift in Newtons', 'Drag in Newtons', 'Velocity in meters per second', ' Pressure in Pascals', 'Boundary velocity in meters per second']
                    ax.set_zlabel(z_label_name_set[index])
                    title_name = surface_names[surface_index] + ' Naca Aerofoil ' + naca_number + ' ' + z_label_name_set[index] + ' at ' + str(alpha) + ' degrees angle of attack.'
                    ax.set_title(title_name)
                    text_x = 2
                    text_y = 14
                    # info_text = "Velocity at inifity is " + str(v_inf) + " meters per second"
                    # info_text = str(info_text)
                    # plt.text(text_x, text_y, info_text, fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
                    # plt.legend()
                    plt.show() 
                    j = j + 4
                index = index + 1
            surface_index = surface_index + 1



    lift_array, drag_array, velocity_array, pressure_array, boundary_array, position_array, lift_coefficient_array, drag_coefficient_array, velocity_coefficient_array, pressure_coefficient_array, boundary_coefficient_array  = Flow_in_xt(velocity_gradient, alpha, x)
    lift_array, drag_array, velocity_array, pressure_array, boundary_array = Resultant_flows(lift_array, drag_array, velocity_array, pressure_array, boundary_array)
    if(plot):
        Plot(time_array, position_array, lift_array, drag_array, velocity_array, pressure_array, boundary_array, alpha)

   
    return lift_array, drag_array, velocity_array, pressure_array, boundary_array, lift_coefficient_array, drag_coefficient_array, velocity_coefficient_array, pressure_coefficient_array, boundary_coefficient_array
        
    
        
        
    
    
# if __name__ == '__main__':
#     windsim(20, 2244, 10, 5, True)
        