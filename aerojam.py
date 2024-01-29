import numpy as np
import matplotlib.pyplot as plt
import sys
from lift_drag_calc import lift_drag_calc
from naca import naca_plot
import subprocess

def aerojam(naca):
    number = str(naca)
    

    naca_plot(number)
    lift_drag_calc()
    # return the data and plot it here 
     
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python my_function_script.py <arg1>")
        sys.exit(1)

    arg1 = str(sys.argv[1])
    result1 = aerojam(arg1)
    print(f"Ran")
