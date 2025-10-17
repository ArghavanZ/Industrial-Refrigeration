# set ROOT, and add src directory to path (The path is this gym_env_ver1 folder!)
import os, sys
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))

# import external packages
import wandb

import numpy as np
import pickle
import argparse
import yaml
import torch
import gymnasium as gym
from typing import Callable


# import stablebaselines
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C ### All support continous actions!
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback)
from wandb.integration.sb3 import WandbCallback 

# import internal packages
#### NOTE: change this import based on which env we want to use!
from gym_env.ref_env import Ref, make_env
from helpers import helper as h
from helpers import device as d





def lr_linear(start: float, end: float) -> Callable[[float], float]:
    # Linearly anneal LR from `start` at begin to `end` at end.
    return lambda progress_remaining: end + (start - end) * progress_remaining

def clip_piecewise(clip_values , progress_values = None) -> Callable[[float], float]:
    # Keep clip=0.2 for first 80% of training, then 0.1 for the rest.
    def f(progress_remaining: float) -> float:
        frac_done = 1.0 - progress_remaining
        if len(clip_values) == 1:
            return float(clip_values[0])
        if len(clip_values) != len(progress_values)+1:
            raise ValueError("clip_values must be one element longer than progress_values")
        
        #### Check where we are in the progress values
        if progress_values is None:
            return float(clip_values[0])

        else:
            for i in range(len(progress_values)-1):
                if frac_done >= progress_values[i] and frac_done < progress_values[i+1]:
                    return float(clip_values[i+1])
            return float(clip_values[0]) if frac_done < progress_values[0] else float(clip_values[-1])
    return f

#### Not tested yet! 
def clip_vf_linear(start: float, end: float) -> Callable[[float], float]:
    # Optional: linearly anneal value-function clip (if you use it)
    return lambda p: end + (start - end) * p


def train(env_cfg, hp_cfg, exp_dir):

    '''
    Main training loop should return model
    '''
    # SETUP ENV
    num_proc = hp_cfg["env_setup"]["num_proc"]  # Number of processes to use
    # fetch env_params from config file
    env_params = env_cfg.copy()
    # Remove unused keys (base params)
    for key in ["name"]:
        env_params["env"].pop(key, None)

    if env_params["reward"]["electricity_price_data"]  != "None":
        # load the electricity price data
        price_path = (f"{ROOT}/{env_params['reward']['electricity_price_data']}")
        env_params["reward"]["electricity_price"] = price_path
        print(f"Loaded electricity price data from {ROOT}/{env_params['reward']['electricity_price_data']}")

    timelimit = int(env_cfg["env"]["Total_time"]/env_cfg["env"]["time_step"])
    # Create the vectorized environment (expand for readability)
    env_list = [
        make_env(
            params = env_params,
            render_mode=None,
            rank=rank,
            seed=hp_cfg["seed"] * 100,  # Guarantee no repeats
            timelimit= timelimit,  # Set the time limit for each environment
        )
        for rank in range(num_proc)
    ]

    set_random_seed(hp_cfg["seed"])
    # Pass the list of callables directly to SubprocVecEnv
    env = SubprocVecEnv(env_list)
    env = VecNormalize(env, norm_obs=hp_cfg["train"]["normalize_observations"], norm_reward=hp_cfg["train"]["reward_normalization"], clip_obs=10.)
    env = VecMonitor(env)

    eval_env = DummyVecEnv([
        make_env(rank=0, params=env_params, render_mode=None, seed=hp_cfg["seed"]*10000, timelimit=timelimit)
    ])
    eval_env = VecNormalize(eval_env, norm_obs=hp_cfg["train"]["normalize_observations"], norm_reward=False, clip_obs=10.)
    eval_env = VecMonitor(eval_env)
    eval_env.obs_rms = env.obs_rms.copy()
    eval_env.ret_rms = env.ret_rms.copy()

    # SETUP MODEL
    algo_name = hp_cfg["model"]["name"]
    raw_arch = hp_cfg["model"]["net_arch"]   # {'pi': [128,128], 'vf': [256,256]} or [256,256]
    activation = hp_cfg["model"].get("activation", "tanh").lower()
    act_cls = torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU



    if isinstance(raw_arch, dict):
    # separate heads -> SB3 expects it wrapped in a list
        net_arch = [dict(
            pi=list(map(int, raw_arch["pi"])),
            vf=list(map(int, raw_arch["vf"]))
        )]
    # else:
    #     net_arch = raw_arch
    elif isinstance(raw_arch, (list, tuple)) and all(isinstance(x, int) for x in raw_arch):
            # shared MLP
        net_arch = list(map(int, raw_arch))
    else:
        raise TypeError("net_arch must be dict{'pi':[..],'vf':[..]} or list[int].")

    policy_kwargs = dict(net_arch=net_arch, activation_fn=act_cls)
    

    ## TODO: add learning rate schedule and clipping schedule for other algorithms if needed

    if algo_name == "PPO":
        lr_schedule = lr_linear(float(hp_cfg["model"]["learning_rate"]), float(hp_cfg["model"]["learning_rate_end"]))
        clip_range = clip_piecewise(hp_cfg["model"]["clip_range"], hp_cfg["model"]["clip_progress"]) 
        # clip_range_vf = clip_vf_linear(1.0, 0.1) if algo_name == "PPO" else None  
        model = PPO(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=lr_schedule,
            n_steps=(hp_cfg["model"]["n_steps"])//num_proc,
            batch_size=hp_cfg["model"]["batch_size"],
            n_epochs=hp_cfg["model"]["n_epochs"],
            gamma=hp_cfg["model"]["gamma"],
            gae_lambda=hp_cfg["model"]["gae_lambda"],
            clip_range=clip_range,
            ent_coef=hp_cfg["model"]["ent_coef"],
            vf_coef=hp_cfg["model"]["vf_coef"],
            max_grad_norm=hp_cfg["model"]["max_grad_norm"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "SAC":
        model = SAC(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            batch_size=hp_cfg["model"]["batch_size"],
            buffer_size=hp_cfg["model"]["buffer_size"],
            learning_starts=hp_cfg["model"]["learning_starts"],
            tau=hp_cfg["model"]["tau"],
            gamma=hp_cfg["model"]["gamma"],
            train_freq=hp_cfg["model"]["train_freq"],
            gradient_steps=hp_cfg["model"]["gradient_steps"],
            ent_coef=hp_cfg["model"]["ent_coef"],
            target_update_interval=hp_cfg["model"]["target_update_interval"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "DDPG":
        model = DDPG(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            batch_size=hp_cfg["model"]["batch_size"],
            buffer_size=hp_cfg["model"]["buffer_size"],
            learning_starts=hp_cfg["model"]["learning_starts"],
            tau=hp_cfg["model"]["tau"],
            gamma=hp_cfg["model"]["gamma"],
            train_freq=hp_cfg["model"]["train_freq"],
            gradient_steps=hp_cfg["model"]["gradient_steps"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "TD3":
        model = TD3(
            policy=hp_cfg["model"]["policy"], 
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            batch_size=hp_cfg["model"]["batch_size"],
            buffer_size=hp_cfg["model"]["buffer_size"],
            learning_starts=hp_cfg["model"]["learning_starts"],
            tau=hp_cfg["model"]["tau"],
            gamma=hp_cfg["model"]["gamma"],
            train_freq=hp_cfg["model"]["train_freq"],
            gradient_steps=hp_cfg["model"]["gradient_steps"],
            target_policy_noise=hp_cfg["model"]["target_policy_noise"],
            target_noise_clip=hp_cfg["model"]["target_noise_clip"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    elif algo_name == "A2C":
        model = A2C(
            policy=hp_cfg["model"]["policy"],
            policy_kwargs=policy_kwargs, 
            env=env,
            learning_rate=float(hp_cfg["model"]["learning_rate"]),
            n_steps=hp_cfg["model"]["n_steps"]//num_proc,
            gamma=hp_cfg["model"]["gamma"],
            gae_lambda=hp_cfg["model"]["gae_lambda"],
            ent_coef=hp_cfg["model"]["ent_coef"],
            vf_coef=hp_cfg["model"]["vf_coef"],
            max_grad_norm=hp_cfg["model"]["max_grad_norm"],
            # rms_prop_eps=hp_cfg["model"]["rms_prop_eps"],
            # use_rms_prop=hp_cfg["model"]["use_rms_prop"],
            tensorboard_log=f"{exp_dir}/tb",  
            verbose=hp_cfg["model"]["verbose"],
            seed=hp_cfg["seed"],
            device=hp_cfg["device"]
        )
    else:
        raise NotImplementedError(f"Algorithm {algo_name} not supported in this script yet.")

    ckpt_cb = CheckpointCallback(
        save_freq=1_000_000, ### Save every 1M steps
        save_path=f"{exp_dir}/checkpoints",
        name_prefix="rl_model",
        save_vecnormalize=True,
        save_replay_buffer=True if algo_name in ["SAC", "DDPG", "TD3"] else False
    )

    eval_cb = EvalCallback(
    eval_env=eval_env,                 # DummyVecEnv(1) with same wrappers as above
    best_model_save_path=f"{exp_dir}/best",
    log_path=f"{exp_dir}/eval_logs",
    eval_freq=500_000,      # 0.5M steps
    n_eval_episodes=20,
    deterministic=True)

    wandb_cb = WandbCallback(
    model_save_path=f"{exp_dir}/models",
    model_save_freq=1000,   # or 0 if you rely only on CheckpointCallback
    gradient_save_freq=0,
    verbose=2)

    callback = CallbackList([ckpt_cb, eval_cb, wandb_cb])
    # Train the model
    model.learn(
        total_timesteps=hp_cfg["algo"]["total_timesteps"], 
        callback=callback,
        log_interval=hp_cfg["algo"]["log_interval"], 
    )

    while hasattr(env, "venv"):
        if isinstance(env, VecNormalize):
            env.save(f"{exp_dir}/vec_normalize.pkl")
            print(f"Saved VecNormalize stats to {exp_dir}/vec_normalize.pkl")
            break
        
        env = env.venv
        

    return None



def get_env_cfg():
    '''
    loads environment param cfg from yaml file path and params provided by args
    '''
    with open(f"{ROOT}/{ARGS.C_path}", 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return env_cfg

def get_hp_cfg():
    '''
    loads  hyperparam cfg from yaml file path and params provided by args
    '''
    with open(f"{ROOT}/{ARGS.HP_path}", 'r') as f:
        hp_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # next we update any meta-parameters that can be set before itearting over seeds
    hp_cfg['project_name'] = f"complex_{ARGS.project_id}"
    return hp_cfg


def run_seed(env_cfg, hp_cfg ,save_dir):
    '''
    runs one instance corresponding to this seed
    NOTE: will skip if this run already exists in our save_dir
    '''
    # create directory for this seed, making sure it does not exist
    exp_dir = f"{save_dir}/{hp_cfg['run_name']}"
    try: 
        os.mkdir(exp_dir)
        os.mkdir(f"{exp_dir}/models")
    except: print(f"WARNING: skipping duplicate run:\n\t{hp_cfg['run_name']}")

    config = dict(env_cfg)
    config.update(hp_cfg)

    # setup wandb
    wandb.init(
        # set the wandb project where this run will be logged
        entity = "Refrigeration", # NOTE: this is your account, change accordingly
        project=hp_cfg['project_name'], # this is our project, defined by env name
        dir=exp_dir, # this is where everything is saved, for now we do not
        name=hp_cfg['run_name'], # this will be used to display our run on wandb (and save here)
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        config = config # this is passed on for tracking the experiment directly (and saved automatically by wandb)
    )
    train(env_cfg, hp_cfg, exp_dir)
    wandb.finish()
    # processing and manual saving can go here
    return None


def main():
    # first we load the env param used for this experiment
    env_cfg = get_env_cfg()

    # next we load the hyperparameters used for this experiment
    hp_cfg_base = get_hp_cfg()
    
    # extract config name for identifying run
    hp_name = ARGS.HP_path.split('/')[-1].split('.')[0]
    param_name = ARGS.C_path.split('/')[-1].split('.')[0]

    # create base directory recursively
    save_dir = f"{ROOT}/{ARGS.result_dir}/{env_cfg['env']['name']}/{param_name}/RL_Model/{hp_cfg_base['model']['name']}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to:\n\t{save_dir}")
    # setup seeding and run all instances iteratively (parallelization should be implemented with bash scripts directly)
    
    for seed in range(ARGS.start_seed, ARGS.start_seed+ARGS.n_seeds):
        # create a config specific to the seed we want to run
        hp_cfg = dict(hp_cfg_base)
        hp_cfg['seed'] = seed
        hp_cfg['run_name'] = f"{param_name}_{hp_name}_{ARGS.run_name}_{seed}" 
        # ideally, we do not return anything and process everything after!
        run_seed(env_cfg, hp_cfg , save_dir)   

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--project_id', type=str, default='Refrigeration', help="additional identiffier used for project name")
    parser.add_argument('--run_name', type=str, default='run', help="all runs saved to the wandb project will use config info ,run_name and seed for identification")
    parser.add_argument('--start_seed', type=int, default=0, help="seed to start from")
    parser.add_argument('--n_seeds', type=int, default=1, help="number of seeds to run")
    parser.add_argument('--HP_path', type=str, default='hyperparameters/H1.yaml', help="location of yaml config file to use for hyperparameters, relative to ROOT dir of project")
    parser.add_argument('--C_path', type=str, default='env_config/C1_1.yaml', help="location of yaml config file to use for environment parameters, relative to ROOT dir of project")
    parser.add_argument('--result_dir', type=str, default='results', help="name of directory where results are stored. Curerntly using env->model to save the result.") ### do not change 
    
    ARGS = parser.parse_args()
    print(ARGS)
    main()