# import necessary libraries
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from utils import bezier_curve
from utils import simulate_airfoil

# creating XFOIL environment
class XFOILEnv(Env):

    def __init__(self):
        self.debug = False
        
        # cleaning up old airfoil files
        foil_num = 0
        available = False
        while available == False:
            if os.path.exists(f"Airfoil_{foil_num}.dat"):
                os.remove(f"Airfoil_{foil_num}.dat")
                foil_num += 1
            else:
                available = True

        # defining action space (continuous Box)
        self.action_space = Box(
            low=np.array([-0.15] * 8, dtype=np.float32),    # max downward movement for 8 control points
            high=np.array([0.15] * 8, dtype=np.float32),    # max upward movement
            dtype=np.float32)

        # defining observation space
        self.observation_space = Box(
            low=np.array([-0.15, -0.15, -0.15, 0.05, -0.15, -0.15, -0.15, -0.15]),
            high=np.array([0.15, 0.15, 0.15, 0.15, -0.05, 0.15, 0.15, 0.15]),
            dtype=np.float32)

        # initial state
        self.state = [0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1]

        # training logs
        self.geometry_log = []
        self.performance_log = []

        
    def step(self, action):
        terminated = True
        truncated = False

        # creating airfoil name
        foil_num = 0
        available = False
        while available == False:
            if os.path.exists(f"Airfoil_{foil_num}.dat"):
                foil_num += 1
            else:
                foil_name = f"Airfoil_{foil_num}"
                available = True

        # clipping action to action space bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # apply the delta y movement to each bezier point
        for i in range(8):
            self.state[i] += action[i]  # update geometry

        # optional: clip back to observation space bounds
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high)

        # generating airfoil
        yu = [0.0, self.state[0], self.state[1], self.state[2], self.state[3], 0.0]
        yl = [0.0, self.state[4], self.state[5], self.state[6], self.state[7], 0.0]
        xl = np.linspace(0, 1, num=len(yl))
        xl[1] = 0.0
        xu = xl[::-1]

        # penalizing invalid geometries
        Pxu, Pyu, Pxl, Pyl = bezier_curve(foil_name, yu, yl)
        Pyu.reverse()  # Reverse in-place; .reverse() returns None
        dy = [0] * len(Pxu)
        for i in range(len(Pxu)):
            dy[i] = Pyu[i] - Pyl[i]
        dy = min(dy)

        d0 = self.state[0] - self.state[7]
        d1 = self.state[1] - self.state[6]
        d2 = self.state[2] - self.state[5]
        d3 = self.state[3] - self.state[4]

        if (dy < 0.0 or 
            d0 < 0.025 or 
            d1 < 0.025 or
            d2 < 0.025 or 
            d3 < 0.025 or
            self.state[3] < 0.025 or 
            self.state[4] > -0.025 or 
            not self.observation_space.contains(np.array(self.state, dtype=np.float32))):
            
            L_D = [0]
            terminated = True
        else:
            L_D = simulate_airfoil(foil_name)

        # calculating reward
        reward = np.mean(L_D)

        # logging geometry and performance
        try:
            self.geometry_log.append({
                "upper_y_points": yu,
                "lower_y_points": yl,
                "upper_x_points": xu,
                "lower_x_points": xl
            })
        except Exception:
            pass

        self.performance_log.append(np.mean(L_D))

        # returning step information
        info = {}
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info
    
        def render(self):
            pass

    def reset(self, seed=None, **kwargs):
        """Reset environment for gymnasium compatibility."""
        self.state = [0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, -0.1]
        return np.array(self.state, dtype=np.float32), {}