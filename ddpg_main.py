import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from custom_env import XFOILEnv
from stable_baselines3 import DDPG
import glob
from utils import plot_airfoil
from utils import plot_performance
 
env = XFOILEnv()

# training SAC model
env.reset()
model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1024,
    batch_size=128,
    tau=0.005,
    gamma=0.95,
    train_freq=(1, "step"),
    gradient_steps=1,
    verbose=1,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    ),
)
model.learn(total_timesteps=4096)
model.save("ddpg_xfoil")
plot_performance(env)
plot_airfoil(env)
