# import necessary libraries
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

# creating XFOIL environment
class XFOILEnv(Env):

    def __init__(self):
        # debug flag: when True, step() will print parameter changes
        self.debug = False
        
        foil_num = 0
        available = False
        while available == False:
            if os.path.exists(f"Airfoil_{foil_num}.dat"):
                os.remove(f"Airfoil_{foil_num}.dat")
                foil_num += 1
            else:
                available = True 

        # defining action space
        self.action_space = Discrete(24) 
        # defining observation space
        # np.array([yu0, yu1, yu2, yl0, yl1, yl2])
        self.observation_space = Box(
            low=np.array([-0.2,-0.2,-0.2,-0.2,-0.3,-0.3,-0.3,-0.3]), 
            high=np.array([0.3,0.3,0.3,0.3,0.2,0.2,0.2,0.2]), 
            dtype=np.float32)
        # initial state
        self.state = [0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, -0.1] 

        
    def step(self, action):
        # creating airfoil name
        foil_num = 0
        available = False
        while available == False:
            if os.path.exists(f"Airfoil_{foil_num}.dat"):
                foil_num += 1
            else:
                foil_name = f"Airfoil_{foil_num}"
                available = True
        
        # updating state based on action
        # Support a larger discrete action space by mapping each action index
        # to (parameter_index, operation). For example, with 8 parameters and
        # 3 ops per parameter (inc, dec, noop) you get 24 actions.
        try:
            n_params = len(self.state)
            ops_per_param = 3
            param_idx = int(action) // ops_per_param
            op = int(action) % ops_per_param
            # step sizes: primary and a smaller step
            step = 0.075
            if 0 <= param_idx < n_params:
                if op == 0:
                    self.state[param_idx] += step
                elif op == 1:
                    self.state[param_idx] -= step
                elif op == 2:
                    # no-op
                    pass
                # debug print if requested and the value actually changed
        except Exception:
            # if action is malformed, ignore and leave state unchanged
            pass


        # generating airfoil
        def bezierCurve(foil_name, yu, yl):
        
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

            with open(f"{foil_name}.dat","w") as file:

                file.write(f"{foil_name}\n")
                for i in range(len(dt)):
                    file.write(f"{Pxu[i]:.5f} {Pyu[i]:.5f}\n")

                for i in range(len(dt)):
                    file.write(f"{Pxl[i]:.5f} {Pyl[i]:.5f}\n")

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

            # return performance metrics
            alpha = polar_data[:,0]
            CL = polar_data[:,1]
            CD = polar_data[:,2]
            L_D = CL/CD
            return L_D

        # generating airfoil
        yu = [0.0, self.state[0], self.state[1], self.state[2], self.state[3], 0.0]
        yl = [0.0, self.state[4], self.state[5], self.state[6], self.state[7], 0.0]
        bezierCurve(foil_name, yu, yl)
        
        # simulating airfoil
        L_D = simulate_airfoil(foil_name)

        # penalizing invalid designs
        d0 = abs(self.state[0]-self.state[4])
        d1 = abs(self.state[1]-self.state[5])
        d2 = abs(self.state[2]-self.state[6])
        d3 = abs(self.state[3]-self.state[7])
        if d0<0.025 or d1<0.025 or d2<0.025 or d3<0.025:
            penalty = -10
        else:
            penalty = 0

        # calculating reward
        reward = np.mean(L_D) + penalty

        # finishing episode
        done = True

        # returning step information
        info = {}
        return self.state, reward, done, info

    def reset(self):

        self.state = [0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, -0.1]
        # self.state = self.observation_space.sample()
        # return the initial observation so env.reset() is not None
        return self.state