import airsim
import numpy as np
import os
from stable_baselines3 import PPO
import torch
from custom_envs.vis_nav_drone_env import VisNavDroneEnv
import networkx as nx
from multiprocessing.managers import SyncManager
from custom_envs.vis_nav_drone_env import get_avail_drone, DRONE_PROXY_AUTH_KEY, \
    DRONE_PROXY_IP_ADDR, DRONE_PROXY_PORT
from stable_baselines3.common.callbacks import BaseCallback
import threading
import time

RUN_FOREVER = True
N_EPISODES = 10000
N_STEPS_PER_EPISODE = 100
MAX_EPISODE_LEN = 10000000000000


class DroneProxyManager(SyncManager):
    pass


def create_drone_proxy_manager(host, port, authkey):
    DroneProxyManager.register('get_drone_proxy', callable=get_avail_drone)
    proxy_manager = DroneProxyManager(address=(host, port), authkey=authkey)

    return proxy_manager


def quadrotor_point_nav(ctxt=None, *, seed=0, gpu=True,load_dir="!"):
    proxy_manager = create_drone_proxy_manager(DRONE_PROXY_IP_ADDR, DRONE_PROXY_PORT, DRONE_PROXY_AUTH_KEY)
    proxy_manager.start()

    env = VisNavDroneEnv(step_length=0.25,
                         image_shape=(144, 256, 1), embed_shape=(144, 256, 1))

    # TODO: setup reinforcement learning model of choice here
    # Recomended platform rllib
    # other alternatives include stable_baselines 3, USC Garage RL, etc.

    # obs = env.reset()
    # while True:
        
    #     action = env.action_space.sample()
    #     obs, reward, done, _ = env.step(action)
    
    #     if done:
    #         obs = env.reset()
    

    from stable_baselines3 import DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    #It will check your custom environment and output additional warnings if needed
    # print(check_env(env))
    # model = DQN(
    # "MultiInputPolicy",
    # env,
    # learning_rate=0.001,
    # verbose=1,
    # batch_size=32,
    # train_freq=4,
    # target_update_interval=1000,
    # learning_starts=1000,
    # buffer_size=50000,
    # max_grad_norm=10,
    # exploration_fraction=0.1,
    # exploration_final_eps=0.01,
    # device="cuda",
    # tensorboard_log="./tb_logs/",
    # )

    
   

    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tb_logs/")

    checkpoint_callback = CheckpointCallback(save_freq=20, save_path='./logs/')
    eval_callback = EvalCallback(env, best_model_save_path='./logs/best_model',
                             log_path='./logs/results', eval_freq=10)


     # Create the callback list
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=1000000, callback=callback_list,reset_num_timesteps=False)
    model.save("ppo_drone_model")

    # # load model
    # model = PPO.load("ppo_drone_model", env, verbose=1)
    # model.learn(total_timesteps=5000)
    # model.save("ppo_drone_model_v2")

     
    