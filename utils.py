import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import sympy as sym
import math
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from matplotlib.animation import FuncAnimation
import glob

def plot_airfoil(env):
    files = sorted(glob.glob("Airfoil_*.dat"), key=lambda x: int(x.split('_')[1].split('.')[0]))

    airfoils = []
    for f in files:
        data = np.loadtxt(f, skiprows=1)  # skip the header
        print(f"Loaded {f}, shape: {data.shape}")
        airfoils.append(data)

    geometry_log = env.geometry_log

    fig, ax = plt.subplots()
    line_airfoil, = ax.plot([], [], 'b-', label='Airfoil Shape')
    line_upper, = ax.plot([], [], 'r-o', label='Upper Bezier')
    line_lower, = ax.plot([], [], 'g-o', label='Lower Bezier')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Airfoil + Bezier Evolution')
    ax.legend()
    ax.grid(True)

    def init():
        line_airfoil.set_data([], [])
        line_upper.set_data([], [])
        line_lower.set_data([], [])
        return line_airfoil, line_upper, line_lower

    def update(frame):
        # Airfoil shape
        airfoil = airfoils[frame]
        line_airfoil.set_data(airfoil[:,0], airfoil[:,1])

        # Bezier curves
        bezier = geometry_log[frame]
        line_upper.set_data(bezier['upper_x_points'], bezier['upper_y_points'])
        line_lower.set_data(bezier['lower_x_points'], bezier['lower_y_points'])

        ax.set_title(f'Airfoil + Bezier Evolution - Step {frame}')
        return line_airfoil, line_upper, line_lower

    frames = min(len(airfoils), len(geometry_log))  # ensure same number of frames
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                        blit=True, interval=5)

    plt.show()

def plot_performance(env, window=50):
    performance_log = np.array(env.performance_log)

    # Compute moving average
    if len(performance_log) >= window:
        smoothed = np.convolve(performance_log, 
                               np.ones(window)/window, 
                               mode='valid')
        x = np.arange(len(smoothed))
    else:
        smoothed = performance_log
        x = np.arange(len(smoothed))

    plt.figure()
    plt.plot(x, smoothed)
    plt.title('Airfoil Performance Over Time (Moving Average)')
    plt.xlabel('Step')
    plt.ylabel('L/D Ratio')
    plt.grid(True)
    plt.show()

def bezier_curve(foil_name, yu, yl):
            xl = np.linspace(0,1,num=len(yl))
            xl[1] = 0.0
            xu = xl[::-1]
            
            t = sym.Symbol('t')
            n = len(yu)-1

            Bxu = 0
            Byu = 0
            Bxl = 0
            Byl = 0

            dt = np.linspace(0,1,num=100)

            for i in range(n+1):
                c = math.factorial(n)/(math.factorial(i)*math.factorial(n-i))
                Bxu += c * (t ** i) * ((1 - t)**(n - i)) * xu[i]
                Byu += c * (t ** i) * ((1 - t)**(n - i)) * yu[i]
                Bxl += c * (t ** i) * ((1 - t)**(n - i)) * xl[i]
                Byl += c * (t ** i) * ((1 - t)**(n - i)) * yl[i]

            Pxu = []
            Pyu = []
            Pxl = []
            Pyl = []

            for i in range(len(dt)):
                Pxu.append(Bxu.subs(t,dt[i]))
                Pyu.append(Byu.subs(t,dt[i]))
                Pxl.append(Bxl.subs(t,dt[i]))
                Pyl.append(Byl.subs(t,dt[i]))
            Pxu.reverse()
            Pyu.reverse()
            Pxl.reverse()
            Pyl.reverse()

            with open(f"{foil_name}.dat","w") as file:

                file.write(f"{foil_name}\n")

                for i in range(len(dt)):
                    file.write(f"{Pxl[i]:.5f} {Pyl[i]:.5f}\n")

                for i in range(len(dt)):
                    file.write(f"{Pxu[i]:.5f} {Pyu[i]:.5f}\n")
                
            return Pxu, Pyu, Pxl, Pyl

def simulate_airfoil(foil_name):
    ai = 5          # angle of attack start
    af = 7         # angle of attack finish
    da = 1        # angle of attack range and step size
    Re = 1000000    # Reynolds number
    nmax = 250      # maximum iterations for XFOIL solver

    # XFOIL input file writer 
    if os.path.exists("polar_file.txt"):
        os.remove("polar_file.txt")
        
    input_file = open("input_file.in", 'w')
    input_file.write("LOAD {0}.dat\n".format(foil_name))
    input_file.write(foil_name + '\n')
    input_file.write("PANE\n")
    input_file.write("OPER\n")
    input_file.write("Visc {0}\n".format(Re))
    input_file.write("PACC\n")
    input_file.write("polar_file.txt\n\n")
    input_file.write("ITER {0}\n".format(nmax))
    input_file.write("ASeq {0} {1} {2}\n".format(ai, af, da))
    input_file.write("\n\n")
    input_file.write("quit\n")
    input_file.close()
    subprocess.call("xfoil.exe < input_file.in", shell=True)
    polar_data = np.loadtxt("polar_file.txt", skiprows=12)

    # Handle both 1D and 2D array results
    if polar_data.ndim == 1:
        polar_data = polar_data.reshape(1, -1)
    
    if polar_data.size == 0:
        return np.array([0.0])  # Return zero L/D if no valid data

    # return performance metrics
    alpha = polar_data[:,0]
    CL = polar_data[:,1]
    CD = polar_data[:,2]
    # Avoid division by zero or negative CD
    L_D = np.divide(CL, CD, where=(CD != 0), out=np.zeros_like(CL))
    return L_D