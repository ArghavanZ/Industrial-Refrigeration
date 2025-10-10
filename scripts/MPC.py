import numpy as np
import yaml 
import matplotlib.pyplot as plt
import casadi as ca 
import do_mpc
import argparse

import os, sys
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))


def load_env_config(ARGS):
    with open(f"{ROOT}/{ARGS.C_path}", 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    return env_cfg



def main():
    env_cfg = load_env_config(ARGS)
    
    # Define the MPC controller
    mpc = do_mpc.controller.MPC(env.model)
    
    setup_mpc = {
        'n_horizon': 10,
        't_step': env.delta_t,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 3,
        'collocation_ni': 2,
        'store_full_solution': True,
        # Add other parameters as needed
    }
    
    mpc.set_param(**setup_mpc)
    
    # Define objective function and constraints
    mterm = ca.mtimes([env.model.x['room_temperatures'].T, env.model.x['room_temperatures']])  # Terminal cost
    lterm = ca.mtimes([env.model.x['room_temperatures'].T, env.model.x['room_temperatures']]) + ca.mtimes([env.model.u['evaporator_actions'].T, env.model.u['evaporator_actions']])  # Stage cost
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    # Set constraints (example: room temperature limits)
    for i in range(env.num_rooms):
        mpc.bounds['lower', '_x', f'room_temperatures[{i}]'] = env.room_min_temps[i]
        mpc.bounds['upper', '_x', f'room_temperatures[{i}]'] = env.room_max_temps[i]
    
    # Control input constraints
    for j in range(env.num_evaporators):
        if env.evaporators[j].continuous:
            mpc.bounds['lower', '_u', f'evaporator_actions[{j}]'] = 0.0
            mpc.bounds['upper', '_u', f'evaporator_actions[{j}]'] = 1.0
        else:
            mpc.bounds['lower', '_u', f'evaporator_actions[{j}]'] = 0
          


    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--C_path', type=str, default='env_config/discrete_joint/C1_2.yaml', help='Path to the environment config file')
    ARGS = parser.parse_args()
    main()