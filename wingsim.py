from windsim import windsim as ws
import numpy as np 
from scipy.integrate import quad
import mpmath
from scipy.integrate import trapezoid
import sympy as sp 
from sympy import re, sqrt, I
import re as regex
import math
import numbers
from scipy.integrate import trapz
from scipy.integrate import simps
from math import prod

# angle of attack 
# lift coefficient 
# drag 
# aspect ratio 
# wing area
# wing loading
# reynolds number 
# mach number 
# critical mach number 
# stall angle 
# flap and slats config 
# sweep angle 
# aileron and elevator defelction 
# boundary layer characteristics 
# wing twist 
# winglets
# wing profile/airfoil shape 

def wing():
    # root_naca, tip_naca, naca_change, num_airfoils winglets=False, alpha, v_inf, time
    # get input values at begining as well as naca airfoils no limit, to number only limited by number of airfoils in sim, 
    print("Welcome to Jamell Ivor/Alvah Samuels' wing designer and aerodynamic simulator.\n\n")
    print("This program will first ask for a series of inputs \nto determine the dimensions of the wing, the NACA numbers of the airfoils and the wind paramters.\n \n ")
    NACA_Numbers = input("Input the NACA airfoils you wish to use as a list/array\nfor example if you want to use 2 NACA airfoils write 2122 3455\nAll NACA airfoils must be 4 digits.\n")
    NACA_Numbers = [int(x) for x in NACA_Numbers.split()]
    Correct = False
    while Correct is False:
        for num in NACA_Numbers:
            if len(str(num)) != 4:
                print(f'Please re-input NACA number {num} as it is not 4 digits')
                index = NACA_Numbers.index(num) 
                NACA_Numbers[index] = input("Please enter the replacement value\n")
            if all(len(str(value)) == 4 for value in NACA_Numbers):
                Correct = True
    print(NACA_Numbers)
    arrays = input("Input x,y,z,t arrays for example 1234 2345 3456 34\n")
  
            
    def calculate_discriminant_product(a, n, x_values):
        if n < 2:
            return 0  # The product is defined for n >= 2
        product = a ** n  # Initialize the product with a^n

        for i in range(n):
            for j in range(i + 1, n):
                product *= (x_values[i] - x_values[j]) ** 2

        return product
    
    def create_list_with_midpoint(n, m):
        if n < 0:
            raise ValueError("n must be a non-negative number")

        if m < 0 or m > n:
            raise ValueError("m must be between 0 and n (inclusive)")

        if n % 2 == 0:
            # If n is even, include m and the integers on either side
            values = list(range(0, m + 1)) + list(range(m, n + 1))
        else:
            # If n is odd, include m and the integers on either side
            values = list(range(0, m + 1)) + list(range(m, n))

        return values
    
    def get_real_solutions(solutions):
        real_solutions = []
        for solution in solutions: 
            real_solutions.append((re(solution)).evalf())
        return real_solutions
    
    def get_coordindates(root_chord, tip_chord, leading_edge, trailing_edge):
        
        xz_triangulation = [[0,0,0], [root_chord, 0, 0], [tip_chord, 0, trailing_edge]]
        xy_triangulation = [[]]
        
        return 
                

# n = 10  # Define the upper range
# m = 5   # Define the midpoint value

# result = create_list_with_midpoint(n, m)
# print(result)
    #  4 equations to generate a wing 
    # x, y, z, t
    def wing_curve(xequation, yequation, zequation, tequation):
        xequation_length = len(xequation)
        yequation_length = len(yequation)
        zequation_length = len(zequation)
        tequation_length = len(tequation)
        xequation_symbols = sp.symbols('x:{}'.format(xequation_length))
        yequation_symbols = sp.symbols('x:{}'.format(yequation_length))
        zequation_symbols = sp.symbols('x:{}'.format(zequation_length))
        tequation_symbols = sp.symbols('x:{}'.format(tequation_length))
        x = sp.symbols('x')
        xequation_coefficients = [0] * xequation_length
        yequation_coefficients = [0] * yequation_length
        zequation_coefficients = [0] * zequation_length
        tequation_coefficients = [0] * tequation_length
        for n in range(0, xequation_length):
            xequation_coefficients[n] = xequation[n]
        for n in range(0, yequation_length):
            yequation_coefficients[n] = yequation[n]
        for n in range(0, zequation_length):
            zequation_coefficients[n] = zequation[n]
        for n in range(0, tequation_length):
            tequation_coefficients[n] = tequation[n]
        xequation_form = [0] * xequation_length
        yequation_form = [0] * yequation_length
        zequation_form = [0] * zequation_length
        tequation_form = [0] * tequation_length
        xequation_coefficients.reverse()
        yequation_coefficients.reverse()
        zequation_coefficients.reverse()
        tequation_coefficients.reverse()
        for n in range(0, xequation_length):
            if n == 0:
                xequation_form[n] = xequation_coefficients[n]
            else: 
                xequation_form[n] = xequation_coefficients[n] * x**(n)
        for n in range(0, yequation_length):
            if n == 0:
                yequation_form[n] = yequation_coefficients[n]
            else: 
                yequation_form[n] = yequation_coefficients[n] * x**(n)
        for n in range(0, zequation_length):
            if n == 0:
                zequation_form[n] = zequation_coefficients[n]
            else: 
                zequation_form[n] = zequation_coefficients[n] * x**(n)
        for n in range(0, tequation_length):
            if n == 0:
                tequation_form[n] = tequation_coefficients[n]
            else: 
                tequation_form[n] = tequation_coefficients[n] * x**(n)
        #  finish it from here basically the same, find the intersections with z for x and y   
        xequation_form = sum(xequation_form)
        yequation_form = sum(yequation_form)
        zequation_form = sum(zequation_form)
        tequation_form = sum(tequation_form)
        
        xequation_solutions = sp.solve(xequation_form, x)
        yequation_solutions = sp.solve(yequation_form, x)
        zequation_solutions = sp.solve(zequation_form, x)
        tequation_solutions = sp.solve(tequation_form, x)
        
        xequation_real_solutions = []
        yequation_real_solutions = []
        zequation_real_solutions = []
        tequation_real_solutions = []
        
        xequation_real_solutions = get_real_solutions(xequation_solutions)
        yequation_real_solutions = get_real_solutions(yequation_solutions)
        zequation_real_solutions = get_real_solutions(zequation_solutions)
        tequation_real_solutions = get_real_solutions(tequation_solutions)
        
        xz_intersection_solutions = sp.solve((xequation_form, zequation_form), x)
        if not xz_intersection_solutions:
            xz_intersection_form = sp.Eq(xequation_form, zequation_form)
            xz_intersection_solutions = sp.solve(xz_intersection_form, x)
        xz_intersection_real_solutions = sorted(get_real_solutions(xz_intersection_solutions))
        
        yz_intersection_solutions = sp.solve((yequation_form, zequation_form), x)
        if not yz_intersection_solutions:
            yz_intersection_form = sp.Eq(yequation_form, zequation_form)
            yz_intersection_solutions = sp.solve(yz_intersection_form, x)
        yz_intersection_real_solutions = sorted(get_real_solutions(yz_intersection_solutions))
        
        root_chord = max([abs(x) for x in xz_intersection_real_solutions])
        tip_chord = min([abs(x) for x in xz_intersection_real_solutions])


        print(f"The root chord length is {root_chord} meters and the tip chord length is {tip_chord} meters")
        # xz intersections values input that and subtract 
        z_x_1 = abs(zequation_form.subs({x: xz_intersection_real_solutions[0]}))
        z_x_2 = abs(zequation_form.subs({x: xz_intersection_real_solutions[-1]}))
        leading_edge = abs(z_x_1 - z_x_2)
        print(f"The leading edge is, {leading_edge} meters\n")
        theta = math.acos((root_chord-tip_chord)/leading_edge)
        trailing_edge = leading_edge*math.sin(theta)
        print(f"The trailing edge is, {trailing_edge} meters\n")
        z_y_1 = abs(zequation_form.subs({x: max(yz_intersection_real_solutions)}))
        z_y_2 = abs(zequation_form.subs({x : min(yz_intersection_real_solutions)}))
        thickness_dist = [z_y_1, z_y_2]
        # get this to make sense with yz 
        zy_i = sp.Eq(zequation_form, yequation_form)
        zy_simp = sp.solve(zy_i, x)
        zy_coefficients = get_real_solutions(zy_simp)
       
        zyequation_form = [0] * len(zy_coefficients)
        for n in range(0, len(zy_coefficients)):
            if n == 0:
                zyequation_form[n] = zy_coefficients[n]
            else: 
                zyequation_form[n] = zy_coefficients[n] * x**(n)
        
        zyequation_form = sum(zyequation_form)
        # print(zyequation_form)
        zy_eq_1 = sp.Eq(zyequation_form, z_y_1)
        zy_eq_2 = sp.Eq(zyequation_form, z_y_2)
        zy_sol_1 = sp.solve(zy_eq_1, x)
        zy_sol_2 = sp.solve(zy_eq_2, x)
        # print(zy_sol_1)
        # print(zy_sol_2)
        num_chord_slices = 10
        zy_x = np.linspace(float(max(zy_sol_1)), float(max(zy_sol_2)), num_chord_slices)
        thickness_range = []
        for i in zy_x:
            result = zyequation_form.subs(x,i)
            thickness_range.append(result)
            
            
         
        print(f"The thickness range between the root and the tip (separated taper ratio) is {thickness_range} meters")
        taper_ratio = min(thickness_dist)/max(thickness_dist)
        print(f"The taper ratio is {taper_ratio}")
        
        
      
        return root_chord, tip_chord, leading_edge, trailing_edge, taper_ratio, thickness_dist
        # aerodynamic center - can not calculate yet 
    
    
       
        # wing twist if any() leading edge twist, distribution, (quarter chord twist - 0.25 from leading edge)
        # flaps and slats - a seperate function
    def wing_twist_equation(distribution_x, distribution_y, distribution_z):

        dist_length_x = len(distribution_x)
        dist_x = sp.symbols('x:{}'.format(dist_length_x))
        dist_xi = [0] * len(distribution_x)
        dist_length_y = len(distribution_y)
        dist_y = sp.symbols('y:{}'.format(dist_length_y))
        dist_yi = [0] * len(distribution_y)
        dist_length_z = len(distribution_z)
        dist_z = sp.symbols('z:{}'.format(dist_length_z))
        dist_zi = [0] * len(distribution_z)
        
            
            # eq_t = sum(dist_x)
        for n in range(0, dist_length_x):
            if n == 0:
                dist_xi[n] = distribution_x[n]
            else:
                dist_xi[n] = distribution_x[n] * dist_x[n]
                
        for n in range(0, dist_length_y):
            if n == 0:
                dist_yi[n] = distribution_y[n]
            else:
                dist_yi[n] = distribution_y[n] * dist_y[n]
                
        for n in range(0, dist_length_z):
            if n == 0:
                dist_zi[n] = distribution_z[n]
            else:
                dist_zi[n] = distribution_z[n] * dist_z[n]
            
        dist_eq_x = sum(dist_xi)
        dist_vars_x = dist_x[1:]
        dist_eq_y = sum(dist_yi)
        dist_vars_y = dist_y[1:]
        dist_eq_z = sum(dist_zi)
        dist_vars_z = dist_z[1:]
        
        twist_eq = [dist_eq_x, dist_vars_x, dist_eq_y, dist_vars_y, dist_eq_z, dist_vars_z]
        return twist_eq
    
    def wing_twist_calculation(twist_eq, position_x_upper, position_x_lower, position_y_upper, position_y_lower, position_z):
        dist_eq_x = twist_eq[0]
        dist_eq_y = twist_eq[2]
        dist_eq_z = twist_eq[4]    
        string_dist_eq_x = str(dist_eq_x)
        string_dist_eq_y = str(dist_eq_y)
        string_dist_eq_z = str(dist_eq_z)
        pattern_x = r'x'
        pattern_y = r'y'        
        pattern_z = r'z'        
        transformed_equation_x = regex.sub(pattern_x, r'x**', string_dist_eq_x)
        transformed_equation_y = regex.sub(pattern_y, r'y**', string_dist_eq_y)
        transformed_equation_z = regex.sub(pattern_z, r'z**', string_dist_eq_z)
        dist_eq_x = sp.sympify(transformed_equation_x)
        dist_eq_y = sp.sympify(transformed_equation_y)
        dist_eq_z = sp.sympify(transformed_equation_z)
        x = sp.symbols('x')
        y = sp.symbols('y')
        z = sp.symbols('z')
        dist_eq_x = dist_eq_x.subs('x', x)
        dist_eq_y = dist_eq_y.subs('y', y)
        dist_eq_z = dist_eq_z.subs('z', z)
        dist_res_root_x_upper = dist_eq_x.subs(x, position_x_upper)
        dist_res_root_x_lower = dist_eq_x.subs(x, position_x_lower)
        dist_res_root_y_upper = dist_eq_y.subs(y, position_y_upper)
        dist_res_root_y_lower = dist_eq_y.subs(y, position_y_lower)
        dist_res_root_z = dist_eq_z.subs(z, position_z)
        twist_angle_x_upper = dist_res_root_x_upper
        twist_angle_x_lower = dist_res_root_x_lower
        twist_angle_y_upper = dist_res_root_y_upper
        twist_angle_y_lower = dist_res_root_y_lower
        twist_angle_z = dist_res_root_z 
        # print(twist_angle_z)
        twist_angle = [twist_angle_x_upper, twist_angle_x_lower, twist_angle_y_upper, twist_angle_y_lower, twist_angle_z]
        return twist_angle
        
    
    def naca_airfoils_attributes(root_naca, tip_naca, leading_edge, trailing_edge,  max_chord, min_chord, thickness, num_airfoils, time, v_inf, naca_change=0.5):
        attributes_array = []
        ratio_lengths = list(range(num_airfoils))
        # Z - along the wing length 
        X_max = max_chord 
        X_min = min_chord 
        X_length = X_max - X_min
        Z_length = trailing_edge
        
        
        
        for n in range(0, num_airfoils):
            if n == 0:
                ratio_lengths[n] = 0
            else:    
                ratio_lengths[n] = 1/n
        cos_theta = (X_length)/Z_length
        theta = math.acos(cos_theta)
        ratio_lengths = sorted(ratio_lengths)

        print(f"The ratio lengths are {ratio_lengths}")
        Y = [0]* num_airfoils
        X = [0] * num_airfoils
        Z = [0] * num_airfoils
        for i in range(0, num_airfoils):
            Z[i] = leading_edge * ratio_lengths[i]
            X[i] = X_max - Z[i]*math.cos(theta)
        X = np.array(X)
        Z = np.array(Z)

       
        # use previous calculations and get 2 x and y's run the calculation with 2 x's and y's using the same distibution 
        #  as the number of airfoils. 
        print(f"The thickness is {thickness}")
        Z = sorted(Z)
        X = sorted(X)
        print(f"These are the X positions {X}")
        print(f"This are the Z positions {Z}")
        M = int(root_naca[0]) / 100  # Maximum camber
        P = int(root_naca[1]) / 10   # Location of maximum camber
        T = int(root_naca[2:]) / 100  # Thickness
        yt = [0] * num_airfoils
        yc = [0] * num_airfoils
        X_upper = [0] * num_airfoils
        X_lower = [0] * num_airfoils
        Y_lower = [0] * num_airfoils
        Y_upper = [0] * num_airfoils
       
        for i in range(0, len(X)):
            yt[i] = 5 * T * (0.2969 * math.sqrt(X[i]) - 0.1260 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4) 
            if P != 0:
                yc[i] = np.where(X[i] < P, (M / P**2) * (2 * P * X[i] - X[i]**2), (M / (1 - P)**2) * (1 - 2 * P + 2 * P * X[i] - X[i]**2))
            else:
                yc[i] = np.zeros_like(X[i])
         # Calculate the upper and lower surfaces' coordinates
            X_upper[i] = X[i] - yt[i] * math.sin(math.atan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * X[i])))
            X_lower[i] = X[i] + yt[i] * math.sin(math.atan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * X[i])))
            Y_upper[i] = yc[i] + yt[i] * math.cos(math.atan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * X[i])))
            Y_lower[i] = yc[i] - yt[i] * math.cos(math.atan((M / (1 - P**2)) * (1 - 2 * P + 2 * P * X[i])))

        # print(f"X upper is {X_upper}")
        # print(f"X lower is {X_lower}")
        # print(f"Y upper is {Y_upper}")
        # print(f"Y lower is {Y_lower}")
        
        list_lift_arrays = [] 
        list_drag_arrays = []
        list_velocity_arrays = []
        list_pressure_arrays = []
        list_boundary_arrays = []
        list_lift_coefficient_array = []

        for i in range(1, num_airfoils):
            # number of airfoils thickness when thickness reaches close to tip thickness change 
            # root chord and root thickness and tip chord and tip thickness with num airfoils being used as the change, 
          
            lift_array, drag_array, velocity_array, pressure_array, boundary_array, lift_coefficient_array, drag_coefficient_array, velocity_coefficient_array, pressure_coefficient_array, boundary_coefficient_array = ws(v_inf, root_naca, alpha, time, X[i], thickness[0]) 
            list_lift_arrays.append(lift_array)
            list_drag_arrays.append(drag_array)
            list_velocity_arrays.append(velocity_array)
            list_pressure_arrays.append(pressure_array)
            list_boundary_arrays.append(boundary_array)
            list_lift_coefficient_array.append(lift_coefficient_array)
            
       
        
        return list_lift_arrays, list_drag_arrays, list_velocity_arrays, list_pressure_arrays, list_boundary_arrays, X, X_upper, X_lower, Y_upper, Y_lower, Z, list_lift_coefficient_array, drag_coefficient_array, velocity_coefficient_array, pressure_coefficient_array, boundary_coefficient_array

    def angle_of_attack(alpha, position, twist_equation):
        X_upper = position[0]
        X_lower = position[1]
        Y_upper = position[2]
        Y_lower = position[3]
        Z = position[4]
        
    # z twist 
        result_twist_eq = wing_twist_equation(twist_equation[0], twist_equation[1], twist_equation[2])
        list_twist_angle_X_upper = []
        list_twist_angle_X_lower = []
        list_twist_angle_Y_upper = []
        list_twist_angle_Y_lower = []
        list_twist_angle_Z = []
        for m in range(0, len(X)):  
            twist = wing_twist_calculation(result_twist_eq, X_upper[m], X_lower[m], Y_upper[m], Y_lower[m], Z[m]) 
            twist_x_upper = twist[0] 
            twist_x_lower = twist[1] 
            twist_y_upper = twist[2] 
            twist_y_lower = twist[3] 
            twist_z = twist[4]
            list_twist_angle_X_upper.append(twist_x_upper)
            list_twist_angle_X_lower.append(twist_x_lower)
            list_twist_angle_Y_upper.append(twist_y_upper)
            list_twist_angle_Y_lower.append(twist_y_lower)
            list_twist_angle_Z.append(twist_z)
        return list_twist_angle_X_upper, list_twist_angle_X_lower, list_twist_angle_Y_upper, list_twist_angle_Y_lower, list_twist_angle_Z
    
        # simulate based on number of airfoils and not length of wing 
            # airfoils don't have a 3d component 
            # trapz points along length and intergrate airfoils with respect to these 
            # poinst along length would be based on number of airfoils 
                 # decompose into left, right, resultant
        # v_inf, naca, alpha, time, plot=False
        #  add all the arrays 
        
        
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

        
        
        
        
    
            
    
    # def flaps_slats():
        
    # def aerodynamic_center(): this is after lift etc
    #     return 

        # only uses function calls 
    root_naca = 2321
    tip_naca = 2222
    num_airfoils = 5
    alpha = 10 
    time = 2 
    v_inf = 10
    root_chord, tip_chord, leading_edge, trailing_edge, taper_ratio, thickness_range = wing_curve([10, 50, 22, 67],[3, 1, 0], [2,3,4,5], [92,34,34])
    list_lift_arrays, list_drag_arrays, list_velocity_arrays, list_pressure_arrays, list_boundary_arrays, X, X_upper, X_lower, Y_upper, Y_lower, Z, list_lift_coefficient_array, drag_coefficient_array, velocity_coefficient_array, pressure_coefficient_array, boundary_coefficient_array = naca_airfoils_attributes("2122", 2244, leading_edge, trailing_edge, root_chord, tip_chord, thickness_range, num_airfoils, alpha, time, v_inf)
    position = [X_upper, X_lower, Y_upper, Y_lower, Z]
    twist_equation = [[0,0.5], [0,0], [-0.5]]
    list_twist_angle_X_upper, list_twist_angle_X_lower, list_twist_angle_Y_upper, list_twist_angle_Y_lower, list_twist_angle_Z = angle_of_attack(alpha, position, twist_equation)
    # time_array, position_array, lift_array, drag_array, velocity_array, pressure_array, boundary_array, alpha
    # Plot(time_array, position_array, list_lift_arrays, list_drag_arrays, list_velocity_arrays, list_pressure_arrays, list_boundary_arrays)
    print(f"The twist in the X upper surface {list_twist_angle_X_upper}")
    print(f"The twist in the X lower surface is {list_twist_angle_X_lower}")
    print(f"The twist in the Y upper surface is {list_twist_angle_Y_upper}")
    print(f"The twist in the Y lower surface is {list_twist_angle_Y_lower}")
    print(f"The twist in the Z angle is {list_twist_angle_Z}")
    # print(list_lift_coefficient_array[-1])
    # sum all the coefficients of lift etc and then graph it 
        
 
    

 
    
    # root_naca, tip_naca, max_chord, min_chord, top_right, bottom_left, num_airfoils, alpha, time, v_inf, naca_change=0.5
    
   
        
            

if __name__ == "__main__":
    wing()
        
        
        
        
        
    
    