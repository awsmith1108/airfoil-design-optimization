import gymnasium as gym
import numpy as np
import matplotlib.pyplot as pltSS
from matplotlib.animation import FuncAnimation
from custom_env import XFOILEnv
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3 import SAC
import glob
from utils import plot_airfoil
from utils import plot_performance
 
env = XFOILEnv()

# training SAC model
env.reset()
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=512,
    batch_size=128,
    tau=0.005,
    gamma=0.95,
    train_freq=(1, "step"),
    gradient_steps=1,
    ent_coef="auto",
    target_update_interval=1,
    use_sde=False,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    ),
    verbose=1
)
model.learn(total_timesteps=4096)
model.save("sac_xfoil")
plot_performance(env)
plot_airfoil(env)
