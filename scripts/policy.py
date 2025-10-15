import os, sys
ROOT = os.path.abspath(os.curdir)
sys.path.append(os.path.abspath(os.path.join(ROOT,'src')))


import json
import numpy as np
import argparse
import yaml
import gymnasium as gym

from stable_baselines3 import PPO, SAC, A2C, DDPG, TD3 # All support continous actions
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from gym_env.ref_env import Ref, make_env
from helpers import helper as h
from helpers import device as d



class base_policy:
    def __init__(self, action_space, action_type, action_mode):
        self.action_space = action_space
        self.action_type = action_type
        self.action_mode = action_mode

    def get_action(self, observation):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def convert_action(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def reset(self):
        pass

############ BASE POLICIES ############

class bang_bang_policy(base_policy):
    def __init__(self, action_space, action_type, action_mode , low_threshold, high_threshold, room_num, evap_num , evap_list ):
        ''' This base policy is a bang bang policy. It will turn off the cooling
             when the room temperature is below the low threshold and on when it is 
            above the high threshold. If the temperature is in the range, it just do nothing. '''
        super().__init__(action_space, action_type, action_mode)
        self.high_threshold = np.asarray(high_threshold) ### high threshold for room temperature
        self.low_threshold = np.asarray(low_threshold) ### low threshold for room temperature
        self.room_num = room_num ### number of rooms
        self.evap_num = evap_num ### number of evaporators
        self.evap_list = evap_list ### list of evaporators' id in each room 
        ### Initialize actions to use as the last action when the policy is reset
        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        else:
            self.evap_actions = None

    def get_action(self, observation):
        
        if "room_temperatures" in observation:
            room_temps = np.asarray(observation["room_temperatures"]).copy()
            

        if "evaporator_control" in self.action_space: ### Only has on/off control of evaporators, does not matter if continuous or discrete
            for i in range(self.room_num):
                if room_temps[i] < self.low_threshold[i]:
                    for evap_id in self.evap_list[i]:
                        self.evap_actions[evap_id] = 0 ### turn off evaporators in the room if room temp is below lower bound
                elif room_temps[i] > self.high_threshold[i]:
                    for evap_id in self.evap_list[i]:
                        self.evap_actions[evap_id] = 1 ### turn on evaporators in the room if room temp is above upper bound

                ### if room temp is within deadband, keep the last action (i.e., do nothing)
        return self.convert_action()


    def convert_action(self): ### Only support evap action 
        action = []
        if self.action_mode == "separate" or self.action_type == "continuous":
            if self.evap_actions is not None:
                action = self.evap_actions.tolist()
        elif self.action_mode == "joint":
            action = 0
            if self.evap_actions is not None:
                for i in range(len(self.evap_actions)):
                    action += self.evap_actions[i] * (2 ** i)
        return action
    

class couple_threshold_policy(base_policy):

    def __init__(self, action_space, action_type, action_mode, low_threshold  , high_threshold, room_num, evap_num , evap_list , evap_seq_list , seq_list):
        ''' This base policy is a coupled policy. It will turn on the cooling for all evaporators of the same group
            when the room temperature of one is above the high threshold and off when the room temperature of one is
        below the low threshold. If the temperature is in the range, it still off. '''

        super().__init__(action_space, action_type, action_mode)
        self.high_threshold = np.asarray(high_threshold) ### high threshold for room temperature
        self.low_threshold = np.asarray(low_threshold) ### low threshold for room temperature
        self.room_num = room_num ### number of rooms
        self.evap_num = evap_num ### number of evaporators
        self.evap_list = evap_list ### list of evaporators' id in each room 
        self.group_list = evap_seq_list ### list of groups for each evaporator
        self.seq_list = seq_list ### list of sequencers in facility 
        
        ### Initialize actions to use as the last action when the policy is reset
        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        else:
            self.evap_actions = None
            
    def get_action(self, observation): 
        
        if "room_temperatures" in observation:
            room_temps = np.asarray(observation["room_temperatures"]).copy()

        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        
        group_on = {group: False for group in self.seq_list} ### initialize all groups to off
        if "evaporator_control" in self.action_space: ### Only has on/off control of evaporators, does not matter if continuous or discrete
            
            for i in range(self.room_num):
                if room_temps[i] > self.high_threshold[i]:
                    for evap_id in self.evap_list[i]: ### check all evaporators in the room
                        self.evap_actions[evap_id] = 1 ### turn on evaporator if room temp is above higher bound
                        group_on[self.group_list[evap_id]] = True ### turn on the group if any evaporator in the group is on

            for group in self.seq_list:
                if group_on[group]: ### if the group is on, turn on all evaporators in the group
                    for list in self.evap_list: ### check the list of evaporators in each room which should belong to the same group
                        if self.group_list[int(list[0])] == group:
                            self.evap_actions[[i for i in list]] = 1

            for i in range(self.room_num):
                if room_temps[i] < self.low_threshold[i]: ### only turns off the evaporator, if the room temp is below the lower bound.
                    for evap_id in self.evap_list[i]:
                        self.evap_actions[evap_id] = 0 ### turn off evaporator if room temp is below lower bound
                
        return self.convert_action()


    def convert_action(self): ### Only support evap action 
        action = []
        if self.action_mode == "separate" or self.action_type == "continuous":
            if self.evap_actions is not None:
                action = self.evap_actions.tolist()
        elif self.action_mode == "joint":
            action = 0
            if self.evap_actions is not None:
                for i in range(len(self.evap_actions)):
                    action += self.evap_actions[i] * (2 ** i)
        return action
    

class couple_policy(base_policy):

    def __init__(self, action_space, action_type, action_mode, low_threshold  , high_threshold, room_num, evap_num , evap_list , evap_seq_list , seq_list):
        ''' This base policy is a coupled policy. It will turn on the cooling for all evaporators of the same group
            when the room temperature of one is above the high threshold and off all the other times '''

        super().__init__(action_space, action_type, action_mode)
        self.high_threshold = np.asarray(high_threshold) ### high threshold for room temperature
        self.low_threshold = np.asarray(low_threshold) ### low threshold for room temperature
        self.room_num = room_num ### number of rooms
        self.evap_num = evap_num ### number of evaporators
        self.evap_list = evap_list ### list of evaporators' id in each room 
        self.group_list = evap_seq_list ### list of groups for each evaporator
        self.seq_list = seq_list ### list of sequences for each evaporator
        
        ### Initialize actions to use as the last action when the policy is reset
        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        else:
            self.evap_actions = None
            

    def get_action(self, observation):  
        if "room_temperatures" in observation:
            room_temps = np.asarray(observation["room_temperatures"]).copy()

        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        
        group_on = {group: False for group in self.seq_list} ### initialize all groups to off
        if "evaporator_control" in self.action_space: ### Only has on/off control of evaporators, does not matter if continuous or discrete
            
            for i in range(self.room_num):
                if room_temps[i] > self.high_threshold[i]:
                    for evap_id in self.evap_list[i]: ### check all evaporators in the room
                        self.evap_actions[evap_id] = 1 ### turn on evaporator if room temp is above higher bound
                        group_on[self.group_list[evap_id]] = True ### turn on the group if any evaporator in the group is on

            for group in self.seq_list:
                if group_on[group]: ### if the group is on, turn on all evaporators in the group
                    for list in self.evap_list: ### check the list of evaporators in each room which should belong to the same group
                        if self.group_list[int(list[0])] == group:
                            self.evap_actions[[i for i in list]] = 1

        return self.convert_action()
    
    def convert_action(self): ### Only support evap action 
        action = []
        if self.action_mode == "separate" or self.action_type == "continuous":
            if self.evap_actions is not None:
                action = self.evap_actions.tolist()
        elif self.action_mode == "joint":
            action = 0
            if self.evap_actions is not None:
                for i in range(len(self.evap_actions)):
                    action += self.evap_actions[i] * (2 ** i)
        return action

########### ELECTRICITY PRICE BASED POLICIES ############

class EP_couple_policy(couple_policy):

    def __init__(self, action_space, action_type, action_mode, low_threshold  , high_threshold, room_num, evap_num , evap_list , evap_seq_list , seq_list, price_offset , lookahead_steps, state_space):
        ''' This base policy is a coupled policy. It will turn on the cooling for all evaporators of the same group
            when the room temperature of one is above the high threshold and off all the other times. Since we have price signal, it will try change the high threshold (take it higher) when the price is about to go lower, 
            and turn on cooling (precooling) when the price is about to go higher
            The timestep to change the policy before the price change can be adjusted is given by the user (lookahead_steps)
            . '''

        super().__init__(action_space, action_type, action_mode, low_threshold, high_threshold, room_num, evap_num, evap_list, evap_seq_list, seq_list)
        self.price_offset = price_offset ### price offset is the temp change we want to make when the price is about to go lower
        self.lookahead_steps = lookahead_steps ### lookahead steps is the number of steps we want to look ahead to see if the price is going to go lower or higher
        if "electricity_price_list" and "EP_remaining" in state_space:
            self.mean_price = None
            self.current_price = 0 ### current price is the current electricity price
        elif "electricity_price" in state_space:
            self.current_price = 0 ### current price is the current electricity price
            self.next_mean_price = 0 ### next mean price is the mean of the electricity prices at the next time step
        

    def get_action(self, observation): 

        if "room_temperatures" in observation:
            room_temps = np.asarray(observation["room_temperatures"])

        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        else:
            self.evap_actions = None

        #################### Check the price signal #################### 
        
        if "electricity_price_list" and "EP_remaining" in observation:
            nz = np.flatnonzero(np.asarray(observation["EP_remaining"] > 0))

            self.current_price = observation["electricity_price_list"][nz[0]]
            if nz[0]+1 >= len(observation["electricity_price_list"]):
                future_price = np.asanyarray([observation["electricity_price_list"][0]]) ### future price is the next price in the list, if there is no next price, it is the first price in the list again
            else:
                future_price = np.asarray(observation["electricity_price_list"][nz[0]+1]) ### future price is the next price in the list

        
            if observation["EP_remaining"][nz[0]] < self.lookahead_steps:
                up = (self.current_price < future_price)
                down = (self.current_price > future_price)
            else: 
                up = False
                down = False

        elif "electricity_price" in observation:
            price_obs = np.asarray(observation["electricity_price_list"])
            self.current_price = price_obs[0] ### current price is the current electricity price
            self.next_mean_price = np.mean(price_obs[1:])

            up = False
            down = False

            ### TODO: Think about the condition for up and down!
            up = (self.current_price + self.current_price/self.lookahead_steps < self.next_mean_price )   
            down = (self.current_price > self.next_mean_price + self.current_price/self.lookahead_steps)


        
        if "evaporator_control" in self.action_space: ### Only has on/off control of evaporators, does not matter if continuous or discrete
            group_on = {group: False for group in self.seq_list} ### initialize all groups to off
            if up: 
                self.evap_actions = np.ones(self.evap_num)  ### turn on all evaporators if the price is going to go higher
            elif down:
                for i in range(self.room_num):
                    if room_temps[i] > self.high_threshold[i] + self.price_offset[i]: ### only turn on the evaporator if the room temp is above the higher bound + price offset
                        for evap_id in self.evap_list[i]: ### check all evaporators in the room
                            self.evap_actions[evap_id] = 1 ### turn on evaporator if room temp is above higher bound + price offset
                            group_on[self.group_list[evap_id]] = True ### turn on the group if any evaporator in the group is on
                for group in self.seq_list:
                    if group_on[group]: ### if the group is on, turn on all evaporators in the group
                        for list in self.evap_list: ### check the list of evaporators in each room which should belong to the same group
                            if self.group_list[int(list[0])] == group:
                                self.evap_actions[[i for i in list]] = 1
                                
            else: ### if the price is not going to change, use the normal coupled policy

                for i in range(self.room_num):
                    if room_temps[i] > self.high_threshold[i]:
                        for evap_id in self.evap_list[i]: ### check all evaporators in the room
                            self.evap_actions[evap_id] = 1 ### turn on evaporator if room temp is above higher bound
                            group_on[self.group_list[evap_id]] = True ### turn on the group if any evaporator in the group is on
                for group in self.seq_list:
                    if group_on[group]: ### if the group is on, turn on all evaporators in the group
                        for list in self.evap_list: ### check the list of evaporators in each room which should belong to the same group
                            if self.group_list[int(list[0])] == group:
                                self.evap_actions[[i for i in list]] = 1

       
        return self.convert_action()



    def convert_action(self):
        action = []
        if self.action_mode == "separate" or self.action_type == "continuous":
            if self.evap_actions is not None:
                action = self.evap_actions.tolist()
        elif self.action_mode == "joint":
            action = 0
            if self.evap_actions is not None:
                for i in range(len(self.evap_actions)):
                    action += self.evap_actions[i] * (2 ** i)
        return action


############ POLICIES WITH SUCTION TEMPERATURE CONTROL ############


class bang_bang_T_suction_policy(bang_bang_policy):


    def __init__(self, action_space, action_type, action_mode , low_threshold, high_threshold ,room_num, evap_num , evap_list , evap_seq_list , seq_list , T_suction_setpoint = None , T_suction_start = None , T_suction_num = None , T_suction_scale = None , T_suction_sp = None ):
        ''' This base policy is a bang bang policy with T_suction control It will turn off the cooling
             when the room temperature is below the low threshold  and on when it is 
            above the high threshold. If the temperature is in the range, it just do nothing (keep previous action) 
            '''

        super().__init__( action_space, action_type, action_mode , low_threshold, high_threshold, room_num, evap_num , evap_list  )
        self.group_list = evap_seq_list ### list of groups for each evaporator
        self.seq_list = seq_list ### list of sequencers (groups) in facility 
        self.set_T_suction_param(T_suction_start , T_suction_num , T_suction_scale , T_suction_sp)

        #### If the setpoint is NONE, we have other forms of control! 
        self.T_suction_setpoint = T_suction_setpoint ### Array of setpoints for suction temperature for each group (bang-bang action) [If we want a fixed setpoint, otherwise it will be None]

        ### Initialize actions to use as the last action when the policy is reset
        if "evaporator_control" in self.action_space:
            self.evap_actions = np.zeros(self.evap_num)  ### initialize all evaporators to off
        else:
            self.evap_actions = None
        

    def set_T_suction_param(self , T_suction_start , T_suction_num , T_suction_scale , T_suction_sp):
        
        if self.action_type == "continuous":
            self.T_suction_sp = np.asarray(T_suction_sp).copy() ### Array of setpoints for suction temperature for each group (continous action)
            self.T_suction_scale = np.asarray(T_suction_scale).copy() ### Array of scale for suction temperature for each group (continous action)
            self.T_suction_start = None ### Array of starting points for suction temperature for each group (discrete action)
            self.T_suction_num = None ### Array of number of discrete points for suction temperature for each group (discrete action)
        else: ### discrete action
            self.T_suction_start = np.asarray(T_suction_start).copy() ### Array of starting points for suction temperature for each group (discrete action)
            self.T_suction_num = np.asarray(T_suction_num).copy() ### Array of number of discrete points for suction temperature for each group (discrete action)
            self.T_suction_sp = None ### Array of setpoints for suction temperature for each group (continous action)
            self.T_suction_scale = None ### Array of scale for suction temperature for each group (continous action)

        if "suction_temperature" in self.action_space:
            self.suction_actions = np.zeros(len(self.seq_list)) ### initialize all suction temperatures to 0
        else:
            self.suction_actions = None


    def get_action(self, observation):
        
        if "room_temperatures" in observation:
            room_temps = np.asarray(observation["room_temperatures"]).copy()
            

        if "evaporator_control" in self.action_space: ### Only has on/off control of evaporators, does not matter if continuous or discrete
            for i in range(self.room_num):
                if room_temps[i] < self.low_threshold[i]:
                    for evap_id in self.evap_list[i]:
                        self.evap_actions[evap_id] = 0 ### turn off evaporators in the room if room temp is below lower bound
                elif room_temps[i] > self.high_threshold[i]:
                    for evap_id in self.evap_list[i]:
                        self.evap_actions[evap_id] = 1 ### turn on evaporators in the room if room temp is above upper bound

                ### if room temp is within deadband, keep the last action (i.e., do nothing)

        if "suction_temperature" in self.action_space:
            if self.action_type == "continuous":
                #### it is T_suction_sp if T_suction_sp is in T_suction_start + i in [0 , T_suction_num -1]
                self.suction_actions =  (self.T_suction_setpoint - self.T_suction_sp) / self.T_suction_scale 
            else: ### discrete action
                for i in range(len(self.T_suction_setpoint)):
                    TS_action_list = self.T_suction_start[i] + np.arange(self.T_suction_num[i])
                    self.suction_actions[i] = TS_action_list[np.argmin(np.abs(TS_action_list - self.T_suction_setpoint[i]))]
                    if self.action_mode == "joint":
                        self.suction_actions = self.suction_actions - self.T_suction_start #### only return the index of the discrete action

        return self.convert_action()
    

    def convert_action(self): ### Only support evap action 
        action = []
        if self.action_mode == "separate" or self.action_type == "continuous":
            if self.evap_actions is not None:
                action = self.evap_actions.tolist()
            if self.suction_actions is not None:
                action.extend(self.suction_actions.tolist())
        elif self.action_mode == "joint":
            action = 0
            if self.evap_actions is not None:
                for i in range(len(self.evap_actions)):
                    action += self.evap_actions[i] * (2 ** i)
            if self.suction_actions is not None:
                ts_action = self.suction_actions[-1]
                for i in range(len(self.suction_actions)-2,-1,-1):
                    ts_action = ts_action * self.T_suction_num[i] + self.suction_actions[i]
                action += ts_action * (2 ** len(self.evap_actions))
        return action
    


class RL_policy:
    def __init__(self, model , algo):
        ''' This base policy is a RL policy. It will take the action as input and return it as output. '''
        norm = os.path.normpath(model)
        models_dir = os.path.dirname(norm)                  
        self.exp_dir = os.path.dirname(models_dir)

        if algo == "PPO":
            self.model = PPO.load(model)
        elif algo == "SAC":
            self.model = SAC.load(model)
        elif algo == "A2C":
            self.model = A2C.load(model)
        elif algo == "DDPG":
            self.model = DDPG.load(model)
        elif algo == "TD3":
            self.model = TD3.load(model)
        else:
            raise ValueError(f"Unknown algorithm: {algo}. Supported algorithms are PPO, SAC, A2C, DDPG, and TD3.")

    def set_env(self,timelimit, env_params):    
        self.dummy_env = DummyVecEnv([lambda: Monitor(gym.wrappers.TimeLimit(Ref(params=env_params), max_episode_steps=timelimit))])
        self.vecnorm = VecNormalize.load(f"{self.exp_dir}/vec_normalize.pkl", self.dummy_env)
        self.vecnorm.training = False
        self.vecnorm.norm_reward = False
        

    def get_action(self, observation):
        obs_in = self.normalize_obs_single(observation)
        action, _states = self.model.predict(obs_in, deterministic=True)
        return action

    def normalize_obs_single(self, obs):
        # Normalize the observation
        if isinstance(obs, dict):
            batch = {k: np.asarray(v, dtype=np.float64)[None, ...] for k, v in obs.items()}
            normed = self.vecnorm.normalize_obs(batch)
            return {k: normed[k][0] for k in normed}  # remove batch dim
        else:
            batch = np.asarray(obs, dtype=np.float64)[None, ...]
            return self.vecnorm.normalize_obs(batch)[0]


############# TRAJECTORY CLASSES ############

class base_Trajectory:
    def __init__(self , policy_name):
        self.policy_name = policy_name
        self.policy = None

    def set_policy(self): ###### make the agent
        self.set_env_params() #### Set the environment parameters to pass to the policy and get trajectory to save data
        self.set_policy_agent() #### Set the policy parameters to pass to make the policy agent and get trajectory to save data

    def get_run_params(self, n_ep, n_seed , start_seed  , data_dir , info_dir , env_params):  ##### set the run parameters 

        self.n_ep = n_ep
        self.n_seed = n_seed
        self.start_seed = start_seed
        self.env_params = env_params
        self.data_dir = data_dir
        self.info_dir = info_dir

    def save_info(self, mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp): #### save the info of the run
        with open(self.info_dir, "w") as f:
            f.write(f"Policy name: {self.policy_name}\n")
            f.write(f"Number of episodes: {self.n_ep}\n")
            f.write(f"Number of seeds: {self.n_seed}\n")
            f.write(f"Start seed: {self.start_seed}\n")
            f.write(f"Environment parameters:\n")
            for key, value in self.env_params.items():
                f.write(f"{key}: {value}\n")

            f.write(f"Evaluation results:\n")
            f.write(f"Mean reward over {self.n_seed*self.n_ep} episodes: {mean_reward}\n")
            f.write(f"Std reward over {self.n_seed*self.n_ep} episodes: {std_reward}\n")
            f.write(f"Violation number for each room over {self.n_seed*self.n_ep} episodes: {violation_num}\n")
            f.write(f"Average on time for each evaporator over {self.n_seed*self.n_ep} episodes: {np.asarray(on_time)*100}\n")
            f.write(f"Total off time for each evaporator over {self.n_seed*self.n_ep} episodes: {off_time}\n")
            f.write(f"Average room temperature for each room over {self.n_seed*self.n_ep} episodes: {Avg_room_temp}\n")


    def get_trajectory(self): ##### Get the trajectory data
        raise NotImplementedError("This method should be overridden by subclasses.")


    def set_env_params(self): #### Set basic parameters to save the data  

        ###### set values for evaluation from env_params!
        delta_t = self.env_params["env"]["time_step"]  # Time step in seconds
        total_t = self.env_params["env"]["Total_time"]  # Total time in seconds
        self.timelimit = int(np.ceil(total_t / delta_t))  # Time limit in time steps
        self.room_num = self.env_params["rooms"]["num_rooms"] # Number of rooms in the facility
        self.evap_num = self.env_params["evaporators"]["num_evaps"] # Number of evaporators in the facility
        self.group_num = len(self.env_params["sequencers"]["seq_list"]) # Number of groups in the facility
        self.compressor_num = np.sum(np.asarray(self.env_params["compressors"]["num_compressors"])) # Number of compressors in the facility
        self.price_num = self.env_params["reward"]["price_num"] ### the length of the electricity price list as observation
        self.action_type = self.env_params["env"]["Action_type"]  # Type of action space (discrete or continuous)
        self.action_space = self.env_params["env"]["Action_space"]  # Number of action spaces (1 for all or 1 per room)
        self.action_mode = self.env_params["env"]["Action_mode"]  # "joint" or "separate" (joint: all actions together, separate: each action separately)
        self.state_space = self.env_params["env"]["state_space"]  # State space representation
        self.min_room_temp = np.asarray(self.env_params["rooms"]["room_min_temp"])  # Minimum room temperature for violation
        self.max_room_temp = np.asarray(self.env_params["rooms"]["room_max_temp"])  # Maximum room temperature for violation

    def set_policy_agent(self): ###  set the policy agent class 
        raise NotImplementedError("This method should be overridden by subclasses.")


class RL_Trajectory(base_Trajectory):
    def __init__(self , policy_name):
        super().__init__(policy_name)

    def set_policy_agent(self):
        
        if self.policy_name == "RL":
            self.policy = RL_policy(self.model , self.algo) ### Set the policy to RL policy
        else:
            raise ValueError(f"Policy name must be 'RL' to set RL parameters. Current policy name is {self.policy_name}.")
        
    def get_policy_params(self, model , algo): #### get the policy parameters (should be called in the main scripts before set_policy)
        self.model = model ### path to the model
        self.algo = algo ### algorithm used to train the model

    ####### Get the run parameters
    def get_run_params(self, n_ep, n_seed , start_seed  , data_dir , info_dir , env_params , hp_params ):

        super().get_run_params(n_ep, n_seed , start_seed  , data_dir , info_dir , env_params)
        self.hp_params = hp_params


    def save_info(self, mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp):
        
        super().save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)

        with open(self.info_dir, "a") as f:
            if self.policy_name == "RL":
                f.write(f"Model: {self.model}\n")
                f.write(f"Algorithm: {self.algo}\n")

            if self.hp_params is not None:
                f.write("Hyperparameters:\n")
                f.write("- model:\n")
                for key, value in self.hp_params['model'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("- environment setup:\n")
                f.write(f" Number of parallel processes: {self.hp_params['env_setup']['num_proc']}\n")
                f.write(f" Timelimit: {self.timelimit}\n")
                f.write(f" total_timesteps: {self.hp_params['algo']['total_timesteps']}\n")


    def get_trajectory(self):

        #### Initialize the dictionary to save trajectory data
        save_dict = {}

        ##### room [ room_num, min_temp, max_temp ]
        if "room_temperatures" in self.state_space:
            room_temps = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
            save_dict["room_temps"] = room_temps

        Q_dist = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
        save_dict["Q_dist"] = Q_dist


        is_violated = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1), dtype=bool)
        save_dict["is_violated"] = is_violated

        ##### evaporator [ evap_num ]
        if "evaporator_utilization" in self.state_space or self.env_params["evaporators"]["util_constraint"] != None:
            on_off_util = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_off_util"] = on_off_util

        if "evaporator_on_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            on_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_time"] = on_time

        if "evaporator_off_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            off_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["off_time"] = off_time

        on_off = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["on_off"] = on_off

        Q_evap = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["Q_evap"] = Q_evap

        ##### electricity price [ price_num ]
        if "electricity_price_list" in self.state_space:
            price_list = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price_list"] = price_list

        if "EP_remaining" in self.state_space:
            EP_remaining = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["EP_remaining"] = EP_remaining

        if "electricity_price" in self.state_space:
            price = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price"] = price  

        ###### Sequencer [ group_num ]
        
        T_suction = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1))
        save_dict["T_suction"] = T_suction

        overloaded = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1), dtype=bool)
        save_dict["overloaded"] = overloaded
        

        ###### Compressors [ compressor_num ]

        W_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max"] = W_max

        W = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W"] = W

        Q_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max"] = Q_max

        Q_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max_clipped"] = Q_max_clipped

        W_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max_clipped"] = W_max_clipped

        ##### Reward and cost (one value per timestep)
        compressor_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["compressor_cost"] = compressor_cost

        violation_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["violation_cost"] = violation_cost

        total_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["total_cost"] = total_cost

        #### One value per episode
        ep_cost = np.zeros(shape=(self.n_seed, self.n_ep))
        save_dict["ep_cost"] = ep_cost
        

        
       
        ############ Run the episodes and collect data for RL! ############
        
        for seed in range(self.start_seed, self.start_seed + self.n_seed):
            
            env = Monitor(gym.wrappers.TimeLimit(Ref(params=self.env_params), max_episode_steps=self.timelimit))
            self.policy.set_env(self.timelimit, self.env_params)

            for ep in range(self.n_ep):
                t = 0
                obs, info = env.reset (seed = 100*seed + ep)
                terminated, truncated = False, False

                #### Room info
                room_temps[seed, ep, :, t] = obs["room_temperatures"]
                is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                Q_dist[seed, ep, :, t] = info["Q_dist"]

                ### Evaporator info
                if "evaporator_utilization" in self.state_space:
                    on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                if "evaporator_on_time" in self.state_space:
                    on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                if "evaporator_off_time" in self.state_space:
                    off_time[seed, ep, :, t] = obs["evaporator_off_time"]
        
                on_off[seed, ep, :, t] = info["evaporator_status"]
                Q_evap[seed, ep, :, t] = info["Q_evap"]
                
                #### Compressor info
                W_max[seed, ep,: , t] = info["W_max"]
                W[seed, ep, :, t] = info["compressor_power"]
                Q_max[seed, ep, :, t] = info["Q_max"]
                W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]
                
                # group info 
                
                T_suction[seed, ep, :, t] = info["T_suction"]
                overloaded[seed, ep, :, t] = info["overloaded"]

                

                while not (terminated or truncated):
                    t += 1
                    action = self.policy.get_action(obs)
                    obs, total_cost[seed, ep, t-1], terminated, truncated, info = env.step(action)

                    #### Room info
                    room_temps[seed, ep, :, t] = obs["room_temperatures"]
                    is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                    Q_dist[seed, ep, :, t] = info["Q_dist"]


                    #### Evaporator info
                    on_off[seed, ep, :, t] = info["evaporator_status"]
                    Q_evap[seed, ep, :, t] = info["Q_evap"]
                    if "evaporator_utilization" in self.state_space:
                        on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                    if "evaporator_on_time" in self.state_space:
                        on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                    if "evaporator_off_time" in self.state_space:
                        off_time[seed, ep, :, t] = obs["evaporator_off_time"]
                    
                    
                    #### Compressor info
                    W_max[seed, ep,: , t] = info["W_max"]
                    W[seed, ep, :, t] = info["compressor_power"]
                    Q_max[seed, ep, :, t] = info["Q_max"]
                    W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                    Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]

                    # group info
                    T_suction[seed, ep, :, t] = info["T_suction"]
                    overloaded[seed, ep, :, t] = info["overloaded"]

                    #### E_price info 
                    if "electricity_price_list" in self.state_space:
                        price_list[seed, ep, :, t-1] = obs["electricity_price_list"]
                        

                    if "EP_remaining" in self.state_space:
                        EP_remaining[seed, ep, :, t-1] = obs["EP_remaining"]

                    if "electricity_price" in self.state_space:
                        price[seed, ep, :, t-1] = obs["electricity_price"]

                    # cost info
                    compressor_cost[seed, ep, t-1] = info["compressor_cost"]
                    violation_cost[seed, ep, t-1] = info["violations_cost"] 

                    ep_cost[seed, ep] += (total_cost[seed, ep, t-1])  # Total cost for the episode 

            env.close()

        mean_reward = np.mean(ep_cost) #Avg reward
        std_reward = np.std(ep_cost) #Std reward (not really reliable)
        violation_num = np.sum(is_violated[:, :, :, :], axis=(0, 1, 3)) #Number of violations for each room
        on_time = np.mean(on_off[:, :, :, :] > 0, axis=(0, 1, 3)) #Average on time for each room (fan speed!)
        off_time = np.sum(on_off[:, :, :, :] == 0, axis=(0, 1, 3)) #Total off time for each room
        Avg_room_temp = np.mean(room_temps[:, :, :, :], axis=(0, 1, 3)) #Average room temperature for each room

        
        numpy_dir = self.data_dir.replace(".csv", ".npz")
        ##### Save result in numpy and compress
        np.savez_compressed(numpy_dir, **{key: save_dict[key] for key in save_dict})

        self.save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)

   
class Simple_Trajectory(base_Trajectory):
    def __init__(self , policy_name):
        super().__init__(policy_name)

    def set_policy_agent(self):

        if self.policy_name == "bang_bang":
            self.policy = bang_bang_policy(self.action_space, self.action_type, self.action_mode , self.low_threshold , self.high_threshold , self.room_num , self.evap_num , self.evap_list)
        elif self.policy_name == "couple":
            self.policy = couple_policy(self.action_space, self.action_type, self.action_mode , self.low_threshold , self.high_threshold , self.room_num , self.evap_num , self.evap_list , self.evap_seq_list , self.seq_list)
        elif self.policy_name == "couple_threshold":
            self.policy = couple_threshold_policy(self.action_space, self.action_type, self.action_mode , self.low_threshold , self.high_threshold ,  self.room_num , self.evap_num  , self.evap_list , self.evap_seq_list , self.seq_list )
        else:
            raise ValueError(f"Unknown policy: {self.policy_name}. Supported policies are bang_bang, couple, couple_threshold")
            

    def get_policy_params(self, low_threshold, high_threshold):
        
        self.high_threshold = np.asarray(high_threshold).copy() ### Array of high temperature thresholds for each room
        self.low_threshold = np.asarray(low_threshold).copy() ### Array of low temperature thresholds for
        
    def set_env_params(self):

        super().set_env_params()
        self.evap_list = (self.env_params["rooms"]["evap_list_id"]) ### list of evaporators in each room
        self.evap_seq_list = (self.env_params["evaporators"]["sequencer"]) ### list of groups for each evaporator
        self.seq_list = (self.env_params["sequencers"]["seq_list"]) ### list of sequencers (groups) in facility


    def save_info(self, mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp):
        super().save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)
        with open(self.info_dir, "a") as f:
            if self.policy_name in ["bang_bang", "couple", "couple_threshold"]:
                f.write(f"High temperature thresholds for each room: {self.high_threshold}\n")
                f.write(f"Low temperature thresholds for each room: {self.low_threshold}\n")

    def get_trajectory(self):

        #### Initialize the dictionary to save trajectory data
        save_dict = {}

        ##### room [ room_num, min_temp, max_temp ]
        if "room_temperatures" in self.state_space:
            room_temps = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
            save_dict["room_temps"] = room_temps

        Q_dist = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
        save_dict["Q_dist"] = Q_dist


        is_violated = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1), dtype=bool)
        save_dict["is_violated"] = is_violated

        ##### evaporator [ evap_num ]
        if "evaporator_utilization" in self.state_space or self.env_params["evaporators"]["util_constraint"] != None:
            on_off_util = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_off_util"] = on_off_util

        if "evaporator_on_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            on_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_time"] = on_time

        if "evaporator_off_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            off_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["off_time"] = off_time

        on_off = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["on_off"] = on_off

        Q_evap = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["Q_evap"] = Q_evap

        ##### electricity price [ price_num ]
        if "electricity_price_list" in self.state_space:
            price_list = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price_list"] = price_list

        if "EP_remaining" in self.state_space:
            EP_remaining = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["EP_remaining"] = EP_remaining

        if "electricity_price" in self.state_space:
            price = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price"] = price  

        ###### Sequencer [ group_num ]

        T_suction = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1))
        save_dict["T_suction"] = T_suction

        overloaded = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1), dtype=bool)
        save_dict["overloaded"] = overloaded
        

        ###### Compressors [ compressor_num ]

        W_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max"] = W_max

        W = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W"] = W

        Q_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max"] = Q_max

        Q_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max_clipped"] = Q_max_clipped

        W_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max_clipped"] = W_max_clipped

        ##### Reward and cost (one value per timestep)
        compressor_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["compressor_cost"] = compressor_cost

        violation_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["violation_cost"] = violation_cost

        total_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["total_cost"] = total_cost

        #### One value per episode
        ep_cost = np.zeros(shape=(self.n_seed, self.n_ep))
        save_dict["ep_cost"] = ep_cost
        

        
       
        ############ Run the episodes and collect data for RL! ############
        
        for seed in range(self.start_seed, self.start_seed + self.n_seed):
            
            env = Monitor(gym.wrappers.TimeLimit(Ref(params=self.env_params), max_episode_steps=self.timelimit))
            for ep in range(self.n_ep):
                t = 0
                obs, info = env.reset (seed = 100*seed + ep)
                terminated, truncated = False, False

                #### Room info
                room_temps[seed, ep, :, t] = obs["room_temperatures"]
                is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                Q_dist[seed, ep, :, t] = info["Q_dist"]

                ### Evaporator info
                if "evaporator_utilization" in self.state_space:
                    on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                if "evaporator_on_time" in self.state_space:
                    on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                if "evaporator_off_time" in self.state_space:
                    off_time[seed, ep, :, t] = obs["evaporator_off_time"]
        
                on_off[seed, ep, :, t] = info["evaporator_status"]
                Q_evap[seed, ep, :, t] = info["Q_evap"]
                
                #### Compressor info
                W_max[seed, ep,: , t] = info["W_max"]
                W[seed, ep, :, t] = info["compressor_power"]
                Q_max[seed, ep, :, t] = info["Q_max"]
                W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]
                
                # group info 
                T_suction[seed, ep, :, t] = info["T_suction"]
                overloaded[seed, ep, :, t] = info["overloaded"]

                

                while not (terminated or truncated):
                    t += 1
                    action = self.policy.get_action(obs)
                    obs, total_cost[seed, ep, t-1], terminated, truncated, info = env.step(action)

                    #### Room info
                    room_temps[seed, ep, :, t] = obs["room_temperatures"]
                    is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                    Q_dist[seed, ep, :, t] = info["Q_dist"]


                    #### Evaporator info
                    on_off[seed, ep, :, t] = info["evaporator_status"]
                    Q_evap[seed, ep, :, t] = info["Q_evap"]
                    if "evaporator_utilization" in self.state_space:
                        on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                    if "evaporator_on_time" in self.state_space:
                        on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                    if "evaporator_off_time" in self.state_space:
                        off_time[seed, ep, :, t] = obs["evaporator_off_time"]
                    
                    
                    #### Compressor info
                    W_max[seed, ep,: , t] = info["W_max"]
                    W[seed, ep, :, t] = info["compressor_power"]
                    Q_max[seed, ep, :, t] = info["Q_max"]
                    W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                    Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]

                    # group info
                    T_suction[seed, ep, :, t] = info["T_suction"]
                    overloaded[seed, ep, :, t] = info["overloaded"]

                    #### E_price info 
                    if "electricity_price_list" in self.state_space:
                        price_list[seed, ep, :, t-1] = obs["electricity_price_list"]
                        

                    if "EP_remaining" in self.state_space:
                        EP_remaining[seed, ep, :, t-1] = obs["EP_remaining"]

                    if "electricity_price" in self.state_space:
                        price[seed, ep, :, t-1] = obs["electricity_price"]

                    # cost info
                    compressor_cost[seed, ep, t-1] = info["compressor_cost"]
                    violation_cost[seed, ep, t-1] = info["violations_cost"] 

                    ep_cost[seed, ep] += (total_cost[seed, ep, t-1])  # Total cost for the episode 

            env.close()

        mean_reward = np.mean(ep_cost) #Avg reward
        std_reward = np.std(ep_cost) #Std reward (not really reliable)
        violation_num = np.sum(is_violated[:, :, :, :], axis=(0, 1, 3)) #Number of violations for each room
        on_time = np.mean(on_off[:, :, :, :] > 0, axis=(0, 1, 3)) #Average on time for each room (fan speed!)
        off_time = np.sum(on_off[:, :, :, :] == 0, axis=(0, 1, 3)) #Total off time for each room
        Avg_room_temp = np.mean(room_temps[:, :, :, :], axis=(0, 1, 3)) #Average room temperature for each room

        
        numpy_dir = self.data_dir.replace(".csv", ".npz")
        ##### Save result in numpy and compress
        np.savez_compressed(numpy_dir, **{key: save_dict[key] for key in save_dict})

        self.save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)

        
class EP_Trajectory(base_Trajectory):
    def __init__(self , policy_name):
        super().__init__(policy_name)

    def set_policy_agent(self):

        if self.policy_name == "EP_couple":
            self.policy = EP_couple_policy(self.action_space, self.action_type, self.action_mode , self.low_threshold , self.high_threshold , self.room_num , self.evap_num , self.evap_list , self.evap_seq_list , self.seq_list ,  self.price_offset , self.lookahead , self.state_space)
        else:
            raise ValueError(f"Unknown policy: {self.policy_name}. Supported policies are EP_couple")

    def get_policy_params(self, low_threshold, high_threshold , price_offset , lookahead):
        
        self.high_threshold = np.asarray(high_threshold).copy() ### Array of high temperature thresholds for each room
        self.low_threshold = np.asarray(low_threshold).copy() ### Array of low temperature thresholds for each room
        self.price_offset = np.asarray(price_offset).copy() ### Array of price offsets for each room
        self.lookahead = lookahead ### lookahead time steps for EP policy

    def set_env_params(self):

        super().set_env_params()
        self.evap_list = (self.env_params["rooms"]["evap_list_id"]) ### list of evaporators in each room
        self.evap_seq_list = (self.env_params["evaporators"]["sequencer"]) ### list of groups for each evaporator
        self.seq_list = (self.env_params["sequencers"]["seq_list"]) ### list of sequencers (groups) in facility

    def save_info(self, mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp):
        super().save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)
        with open(self.info_dir, "a") as f:
            if self.policy_name in ["Ep_couple"]:
                f.write(f"Lookahead time steps: {self.lookahead}\n")
                f.write(f"High temperature thresholds for each room: {self.high_threshold}\n")
                f.write(f"Low temperature thresholds for each room: {self.low_threshold}\n")
                f.write(f"Price offsets for each room: {self.price_offset}\n")
                f.write(f"High temperature thresholds for each room when the price is about to decrease: {self.high_threshold + self.price_offset}\n")


    def get_trajectory(self):

        #### Initialize the dictionary to save trajectory data
        save_dict = {}

        ##### room [ room_num, min_temp, max_temp ]
        if "room_temperatures" in self.state_space:
            room_temps = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
            save_dict["room_temps"] = room_temps

        Q_dist = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
        save_dict["Q_dist"] = Q_dist


        is_violated = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1), dtype=bool)
        save_dict["is_violated"] = is_violated

        ##### evaporator [ evap_num ]
        if "evaporator_utilization" in self.state_space or self.env_params["evaporators"]["util_constraint"] != None:
            on_off_util = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_off_util"] = on_off_util

        if "evaporator_on_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            on_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_time"] = on_time

        if "evaporator_off_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            off_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["off_time"] = off_time

        on_off = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["on_off"] = on_off

        Q_evap = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["Q_evap"] = Q_evap

        ##### electricity price [ price_num ]
        if "electricity_price_list" in self.state_space:
            price_list = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price_list"] = price_list

        if "EP_remaining" in self.state_space:
            EP_remaining = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["EP_remaining"] = EP_remaining

        if "electricity_price" in self.state_space:
            price = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price"] = price  

        ###### Sequencer [ group_num ]

        T_suction = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1))
        save_dict["T_suction"] = T_suction

        overloaded = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1), dtype=bool)
        save_dict["overloaded"] = overloaded
        

        ###### Compressors [ compressor_num ]

        W_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max"] = W_max

        W = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W"] = W

        Q_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max"] = Q_max

        Q_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max_clipped"] = Q_max_clipped

        W_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max_clipped"] = W_max_clipped

        ##### Reward and cost (one value per timestep)
        compressor_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["compressor_cost"] = compressor_cost

        violation_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["violation_cost"] = violation_cost

        total_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["total_cost"] = total_cost

        #### One value per episode
        ep_cost = np.zeros(shape=(self.n_seed, self.n_ep))
        save_dict["ep_cost"] = ep_cost
        

        
       
        ############ Run the episodes and collect data for RL! ############
        
        for seed in range(self.start_seed, self.start_seed + self.n_seed):
            
            env = Monitor(gym.wrappers.TimeLimit(Ref(params=self.env_params), max_episode_steps=self.timelimit))
            for ep in range(self.n_ep):
                t = 0
                obs, info = env.reset (seed = 100*seed + ep)
                terminated, truncated = False, False

                #### Room info
                room_temps[seed, ep, :, t] = obs["room_temperatures"]
                is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                Q_dist[seed, ep, :, t] = info["Q_dist"]

                ### Evaporator info
                if "evaporator_utilization" in self.state_space:
                    on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                if "evaporator_on_time" in self.state_space:
                    on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                if "evaporator_off_time" in self.state_space:
                    off_time[seed, ep, :, t] = obs["evaporator_off_time"]
        
                on_off[seed, ep, :, t] = info["evaporator_status"]
                Q_evap[seed, ep, :, t] = info["Q_evap"]
                
                #### Compressor info
                W_max[seed, ep,: , t] = info["W_max"]
                W[seed, ep, :, t] = info["compressor_power"]
                Q_max[seed, ep, :, t] = info["Q_max"]
                W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]
                
                # group info 
                T_suction[seed, ep, :, t] = info["T_suction"]
                overloaded[seed, ep, :, t] = info["overloaded"]

                

                while not (terminated or truncated):
                    t += 1
                    action = self.policy.get_action(obs)
                    obs, total_cost[seed, ep, t-1], terminated, truncated, info = env.step(action)

                    #### Room info
                    room_temps[seed, ep, :, t] = obs["room_temperatures"]
                    is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                    Q_dist[seed, ep, :, t] = info["Q_dist"]


                    #### Evaporator info
                    on_off[seed, ep, :, t] = info["evaporator_status"]
                    Q_evap[seed, ep, :, t] = info["Q_evap"]
                    if "evaporator_utilization" in self.state_space:
                        on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                    if "evaporator_on_time" in self.state_space:
                        on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                    if "evaporator_off_time" in self.state_space:
                        off_time[seed, ep, :, t] = obs["evaporator_off_time"]
                    
                    
                    #### Compressor info
                    W_max[seed, ep,: , t] = info["W_max"]
                    W[seed, ep, :, t] = info["compressor_power"]
                    Q_max[seed, ep, :, t] = info["Q_max"]
                    W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                    Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]

                    # group info
                    T_suction[seed, ep, :, t] = info["T_suction"]
                    overloaded[seed, ep, :, t] = info["overloaded"]

                    #### E_price info 
                    if "electricity_price_list" in self.state_space:
                        price_list[seed, ep, :, t-1] = obs["electricity_price_list"]
                        

                    if "EP_remaining" in self.state_space:
                        EP_remaining[seed, ep, :, t-1] = obs["EP_remaining"]

                    if "electricity_price" in self.state_space:
                        price[seed, ep, :, t-1] = obs["electricity_price"]

                    # cost info
                    compressor_cost[seed, ep, t-1] = info["compressor_cost"]
                    violation_cost[seed, ep, t-1] = info["violations_cost"] 

                    ep_cost[seed, ep] += (total_cost[seed, ep, t-1])  # Total cost for the episode 

            env.close()

        mean_reward = np.mean(ep_cost) #Avg reward
        std_reward = np.std(ep_cost) #Std reward (not really reliable)
        violation_num = np.sum(is_violated[:, :, :, :], axis=(0, 1, 3)) #Number of violations for each room
        on_time = np.mean(on_off[:, :, :, :] > 0, axis=(0, 1, 3)) #Average on time for each room (fan speed!)
        off_time = np.sum(on_off[:, :, :, :] == 0, axis=(0, 1, 3)) #Total off time for each room
        Avg_room_temp = np.mean(room_temps[:, :, :, :], axis=(0, 1, 3)) #Average room temperature for each room

        
        numpy_dir = self.data_dir.replace(".csv", ".npz")
        ##### Save result in numpy and compress
        np.savez_compressed(numpy_dir, **{key: save_dict[key] for key in save_dict})

        self.save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)
           
#### TODO: think about the config file and setting T_suction parameters more (There is a T_suction_sp when discrete action which is not making any sense now) 
class Tsuction_Trajectory(base_Trajectory): 
    def __init__(self , policy_name):
        super().__init__(policy_name)

    def set_policy_agent(self):

        if self.policy_name == "bang_bang_T_suction":
            self.policy = bang_bang_T_suction_policy(self.action_space, self.action_type, self.action_mode , self.low_threshold , self.high_threshold , self.room_num , self.evap_num , self.evap_list , self.evap_seq_list , self.seq_list , self.T_suction_setpoint , self.T_suction_start , self.T_suction_num , self.T_suction_scale , self.T_suction_sp)
        else:
            raise ValueError(f"Unknown policy: {self.policy_name}. Supported policies are bang_bang_T_suction")
            

    def get_policy_params(self, low_threshold, high_threshold , T_suction_setpoint):
        
        self.high_threshold = np.asarray(high_threshold).copy() ### Array of high temperature thresholds for each room
        self.low_threshold = np.asarray(low_threshold).copy() ### Array of low temperature thresholds for each room
        self.T_suction_setpoint = np.asarray(T_suction_setpoint).copy() ### Array of setpoints for suction temperature for each group (bang-bang action) [If we want a fixed setpoint, otherwise it will be None]
        

    def set_env_params(self):

        super().set_env_params()
        self.evap_list = (self.env_params["rooms"]["evap_list_id"]) ### list of evaporators in each room
        self.evap_seq_list = (self.env_params["evaporators"]["sequencer"]) ### list of groups for each evaporator
        self.seq_list = (self.env_params["sequencers"]["seq_list"]) ### list of sequencers (groups) in facility
        if "suction_temperature" in self.action_space:
            if self.action_type == "continuous":
                self.T_suction_sp = np.asarray(self.env_params["sequencers"]["T_suction_sp"]).copy() ### Array of setpoints for suction temperature for each group (continous action)
                self.T_suction_scale = np.asarray(self.env_params["sequencers"]["T_suction_scale"]).copy() ### Array of scale for suction temperature for each group (continous action)
                self.T_suction_start = None ### Array of starting points for suction temperature for each group (discrete action)
                self.T_suction_num = None ### Array of number of discrete points for suction temperature for each group (discrete action)
            else: ### discrete action
                self.T_suction_start = np.asarray(self.env_params["sequencers"]["T_suction_start"]).copy() ### Array of starting points for suction temperature for each group (discrete action)
                self.T_suction_num = np.asarray(self.env_params["sequencers"]["T_suction_num"]).copy() ### Array of number of discrete points for suction temperature for each group (discrete action)
                self.T_suction_sp = None ### Array of setpoints for suction temperature for each group (continous action)
                self.T_suction_scale = None ### Array of scale for suction temperature for each group (continous action)
        else:
            raise ValueError("suction_temperature must be in action_space for bang_bang_T_suction policy.")
        

    def save_info(self, mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp):
        super().save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)
        with open(self.info_dir, "a") as f:
            if self.policy_name in ["bang_bang_T_suction"]:
                f.write(f"High temperature thresholds for each room: {self.high_threshold}\n")
                f.write(f"Low temperature thresholds for each room: {self.low_threshold}\n")
                f.write(f"Suction temperature setpoints for each group: {self.T_suction_setpoint}\n")
    
    def get_trajectory(self):

        #### Initialize the dictionary to save trajectory data
        save_dict = {}

        ##### room [ room_num, min_temp, max_temp ]
        if "room_temperatures" in self.state_space:
            room_temps = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
            save_dict["room_temps"] = room_temps

        Q_dist = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1))
        save_dict["Q_dist"] = Q_dist


        is_violated = np.zeros(shape=(self.n_seed, self.n_ep, self.room_num, self.timelimit+1), dtype=bool)
        save_dict["is_violated"] = is_violated

        ##### evaporator [ evap_num ]
        if "evaporator_utilization" in self.state_space or self.env_params["evaporators"]["util_constraint"] != None:
            on_off_util = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_off_util"] = on_off_util

        if "evaporator_on_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            on_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["on_time"] = on_time

        if "evaporator_off_time" in self.state_space or self.env_params["evaporators"]["on_off_min_constraint"] != None :
            off_time = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
            save_dict["off_time"] = off_time

        on_off = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["on_off"] = on_off

        Q_evap = np.zeros(shape=(self.n_seed, self.n_ep, self.evap_num, self.timelimit+1))
        save_dict["Q_evap"] = Q_evap

        ##### electricity price [ price_num ]
        if "electricity_price_list" in self.state_space:
            price_list = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price_list"] = price_list

        if "EP_remaining" in self.state_space:
            EP_remaining = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["EP_remaining"] = EP_remaining

        if "electricity_price" in self.state_space:
            price = np.zeros(shape=(self.n_seed, self.n_ep, self.price_num, self.timelimit))
            save_dict["price"] = price  

        ###### Sequencer [ group_num ]

        T_suction = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1))
        save_dict["T_suction"] = T_suction

        overloaded = np.zeros(shape=(self.n_seed, self.n_ep, self.group_num, self.timelimit+1), dtype=bool)
        save_dict["overloaded"] = overloaded
        

        ###### Compressors [ compressor_num ]

        W_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max"] = W_max

        W = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W"] = W

        Q_max = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max"] = Q_max

        Q_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["Q_max_clipped"] = Q_max_clipped

        W_max_clipped = np.zeros(shape=(self.n_seed, self.n_ep, self.compressor_num, self.timelimit+1))
        save_dict["W_max_clipped"] = W_max_clipped

        ##### Reward and cost (one value per timestep)
        compressor_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["compressor_cost"] = compressor_cost

        violation_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["violation_cost"] = violation_cost

        total_cost = np.zeros(shape=(self.n_seed, self.n_ep, self.timelimit))
        save_dict["total_cost"] = total_cost

        #### One value per episode
        ep_cost = np.zeros(shape=(self.n_seed, self.n_ep))
        save_dict["ep_cost"] = ep_cost
        

        
       
        ############ Run the episodes and collect data for RL! ############
        
        for seed in range(self.start_seed, self.start_seed + self.n_seed):
            
            env = Monitor(gym.wrappers.TimeLimit(Ref(params=self.env_params), max_episode_steps=self.timelimit))
            self.policy.set_env(self.timelimit, self.env_params)

            for ep in range(self.n_ep):
                t = 0
                obs, info = env.reset (seed = 100*seed + ep)
                terminated, truncated = False, False

                #### Room info
                room_temps[seed, ep, :, t] = obs["room_temperatures"]
                is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                Q_dist[seed, ep, :, t] = info["Q_dist"]

                ### Evaporator info
                if "evaporator_utilization" in self.state_space:
                    on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                if "evaporator_on_time" in self.state_space:
                    on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                if "evaporator_off_time" in self.state_space:
                    off_time[seed, ep, :, t] = obs["evaporator_off_time"]
        
                on_off[seed, ep, :, t] = info["evaporator_status"]
                Q_evap[seed, ep, :, t] = info["Q_evap"]
                
                #### Compressor info
                W_max[seed, ep,: , t] = info["W_max"]
                W[seed, ep, :, t] = info["compressor_power"]
                Q_max[seed, ep, :, t] = info["Q_max"]
                W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]
                
                # group info 
                T_suction[seed, ep, :, t] = info["T_suction"]
                overloaded[seed, ep, :, t] = info["overloaded"]

                

                while not (terminated or truncated):
                    t += 1
                    action = self.policy.get_action(obs)
                    obs, total_cost[seed, ep, t-1], terminated, truncated, info = env.step(action)

                    #### Room info
                    room_temps[seed, ep, :, t] = obs["room_temperatures"]
                    is_violated[seed, ep, :, t] = (room_temps[seed, ep, :, t] > self.max_room_temp[:]) | (room_temps[seed, ep, :, t] < self.min_room_temp[:])
                    Q_dist[seed, ep, :, t] = info["Q_dist"]


                    #### Evaporator info
                    on_off[seed, ep, :, t] = info["evaporator_status"]
                    Q_evap[seed, ep, :, t] = info["Q_evap"]
                    if "evaporator_utilization" in self.state_space:
                        on_off_util[seed, ep, :, t] = obs["evaporator_utilization"]
                    if "evaporator_on_time" in self.state_space:
                        on_time[seed, ep, :, t] = obs["evaporator_on_time"]
                    if "evaporator_off_time" in self.state_space:
                        off_time[seed, ep, :, t] = obs["evaporator_off_time"]
                    
                    
                    #### Compressor info
                    W_max[seed, ep,: , t] = info["W_max"]
                    W[seed, ep, :, t] = info["compressor_power"]
                    Q_max[seed, ep, :, t] = info["Q_max"]
                    W_max_clipped[seed, ep, :, t] = info["W_max_clip"]
                    Q_max_clipped[seed, ep, :, t] = info["Q_max_clip"]

                    # group info
                    T_suction[seed, ep, :, t] = info["T_suction"]
                    overloaded[seed, ep, :, t] = info["overloaded"]

                    #### E_price info 
                    if "electricity_price_list" in self.state_space:
                        price_list[seed, ep, :, t-1] = obs["electricity_price_list"]
                        

                    if "EP_remaining" in self.state_space:
                        EP_remaining[seed, ep, :, t-1] = obs["EP_remaining"]

                    if "electricity_price" in self.state_space:
                        price[seed, ep, :, t-1] = obs["electricity_price"]

                    # cost info
                    compressor_cost[seed, ep, t-1] = info["compressor_cost"]
                    violation_cost[seed, ep, t-1] = info["violations_cost"] 

                    ep_cost[seed, ep] += (total_cost[seed, ep, t-1])  # Total cost for the episode 

            env.close()

        mean_reward = np.mean(ep_cost) #Avg reward
        std_reward = np.std(ep_cost) #Std reward (not really reliable)
        violation_num = np.sum(is_violated[:, :, :, :], axis=(0, 1, 3)) #Number of violations for each room
        on_time = np.mean(on_off[:, :, :, :] > 0, axis=(0, 1, 3)) #Average on time for each room (fan speed!)
        off_time = np.sum(on_off[:, :, :, :] == 0, axis=(0, 1, 3)) #Total off time for each room
        Avg_room_temp = np.mean(room_temps[:, :, :, :], axis=(0, 1, 3)) #Average room temperature for each room

        
        numpy_dir = self.data_dir.replace(".csv", ".npz")
        ##### Save result in numpy and compress
        np.savez_compressed(numpy_dir, **{key: save_dict[key] for key in save_dict})

        self.save_info(mean_reward, std_reward, violation_num, on_time, off_time, Avg_room_temp)



def main():

    #### Load configurations of environment
    with open(f"{ROOT}/{ARGS.C_path}", 'r') as f:
        env_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    
    #### Load configurations of policy evaluayion
    with open(f"{ROOT}/{ARGS.E_path}", 'r') as f:
        policy_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    if  policy_cfg["policy"]["name"] == "RL":
    #### Load configurations of hyperparameters 
        with open(f"{ROOT}/{ARGS.HP_path}", 'r') as f:
            hp_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        hp_name = ARGS.HP_path.split('/')[-1].split('.')[0]
        algo = hp_cfg['model']['name']
            
    else:
        hp_cfg = None
        


    
    #### set run parameters  

    policy_name = policy_cfg["policy"]["name"] #### What policy do we want to use
    policy_type = policy_cfg["policy"]["type"] #### What type of trajecrory class do we need to use
    
    n_ep = policy_cfg["run"]["n_episodes"] #### Number of episodes to run
    n_seed = policy_cfg["run"]["n_seeds"] #### Number of different seeds to
    start_seed = policy_cfg["run"]["start_seed"] #### Starting seed
    save = policy_cfg["run"]["save"] #### Save mode (csv or numpy)


    if policy_name == "RL":
        model= policy_cfg["policy"]["model_name"] #### Model name
        model_mode = policy_cfg["policy"]["model_mode"] #### Model mode
        mode_timesteps = policy_cfg["policy"]["mode_timestep"] #### Model timesteps (only if model_mode is checkpoints)


    if policy_name in ["bang_bang", "couple","couple_threshold"]:
        low_threshold = policy_cfg["policy"]["low_threshold"] #### Low temperature thresholds for each room
        high_threshold = policy_cfg["policy"]["high_threshold"] #### High temperature thresholds for each room

    if policy_name in ["EP_couple"]:
        low_threshold = policy_cfg["policy"]["low_threshold"] #### Low temperature thresholds for each room
        high_threshold = policy_cfg["policy"]["high_threshold"] #### High temperature thresholds for each room
        price_offset = policy_cfg["policy"]["price_offset"] #### Price offset for each room
        lookahead = policy_cfg["policy"]["lookahead"] #### Lookahead time steps

    if policy_name in ["bang_bang_T_suction"]:
        low_threshold = policy_cfg["policy"]["low_threshold"] #### Low temperature thresholds for each room
        high_threshold = policy_cfg["policy"]["high_threshold"] #### High temperature thresholds for each room
        T_suction_setpoint = policy_cfg["policy"]["T_suction_sp"]
        
    ##### LOAD EXTRA DATA FOR ENVIRONMENT IF NEEDED
    if env_cfg["reward"]["electricity_price_data"]  != "None":
        # load the electricity price data
        price_path = (f"{ROOT}/{env_cfg['reward']['electricity_price_data']}")
        env_cfg["reward"]["electricity_price"] = price_path
        print(f"Loaded electricity price data from {ROOT}/{env_cfg['reward']['electricity_price_data']}")

    c_name = ARGS.C_path.split("/")[-1].split(".yaml")[0] ### get config name

    if policy_name in ["bang_bang", "couple","couple_threshold" , "bang_bang_T_suction" , "EP_couple"]:
        room_num = env_cfg["rooms"]["num_rooms"]
        group_num = len(env_cfg["sequencers"]["seq_list"])
        if len(high_threshold) != room_num:
            raise ValueError(f"Number of room setpoints ({len(high_threshold)}) does not match number of rooms ({room_num}) in environment configuration.")
        if len(low_threshold) != room_num:
            raise ValueError(f"Number of low thresholds ({len(low_threshold)}) does not match number of rooms ({room_num}) in environment configuration.")

    if policy_name in ["EP_couple"]:
        if len(price_offset) != room_num:
            raise ValueError(f"Number of price offsets ({len(price_offset)}) does not match number of rooms ({room_num}) in environment configuration.")
        if lookahead <=0:
            raise ValueError(f"Lookahead must be a positive integer. Current lookahead is {lookahead}.")
        

    if policy_name in ["bang_bang_T_suction"]:
        if "suction_temperature" in env_cfg["env"]["Action_space"]:
            if len(T_suction_setpoint) != group_num:
                raise ValueError(f"Number of suction setpoint temperatures ({len(T_suction_setpoint)}) does not match number of groups ({group_num}) in environment configuration.")
        else:
            raise ValueError("suction_temperature must be in action_space for bang_bang_T_suction policy.")
        

    save_dir = f"{ROOT}/results/{env_cfg['env']['name']}/{c_name}/test_data/{policy_name}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to:\n\t{save_dir}")

    ###### Form RL POLICY PATH
    if policy_name == "RL":
        if model_mode == "best":
            model_dir = f"{ROOT}/results/{env_cfg['env']['name']}/{c_name}/RL_Model/{algo}/{model}/{model_mode}/best_model.zip"
        elif model_mode == "checkpoints":
            model_dir = f"{ROOT}/results/{env_cfg['env']['name']}/{c_name}/RL_Model/{algo}/{model}/{model_mode}/rl_model_{mode_timesteps}_steps.zip"
        elif model_mode == "models":
            model_dir = f"{ROOT}/results/{env_cfg['env']['name']}/{c_name}/RL_Model/{algo}/{model}/{model_mode}/model.zip"
        else:
            raise ValueError(f"Unknown model mode: {model_mode}. Supported modes are best, checkpoints, models.")
    
    if policy_name == "RL" and not os.path.exists(model_dir):
        raise ValueError(f"Model file does not exist: {model_dir}")

    if policy_name == "RL":
        data_dir = f"{save_dir}/{model}/{model_mode}/trajectories_0.csv"
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        info_dir = f"{save_dir}/{model}/{model_mode}/info.txt"
        os.makedirs(os.path.dirname(info_dir), exist_ok=True)
        if os.path.exists(f"{save_dir}/{model}/{model_mode}"):
            base_csv = f"{save_dir}/{model}/{model_mode}/trajectories"
            base_txt = f"{save_dir}/{model}/{model_mode}/info"
            #### get additional name id
            new_id = 0
            while os.path.exists(f"{base_csv}_{new_id}.csv") or os.path.exists(f"{base_txt}_{new_id}.txt"):
                new_id += 1
            # Create a new file
            data_dir = f"{base_csv}_{new_id}.csv"
            info_dir = f"{base_txt}_{new_id}.txt"
            print(f"Creating a new file: {data_dir} and {info_dir}")
    
    else:
        data_dir = f"{save_dir}/trajectories_0.csv"
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        info_dir = f"{save_dir}/info.txt"
        os.makedirs(os.path.dirname(info_dir), exist_ok=True)
        if os.path.exists(save_dir):
            base_csv = f"{save_dir}/trajectories"
            base_txt = f"{save_dir}/info"
            #### get additional name id
            new_id = 0
            while os.path.exists(f"{base_csv}_{new_id}.csv") or os.path.exists(f"{base_txt}_{new_id}.txt"):
                new_id += 1
            # Create a new file
            data_dir = f"{base_csv}_{new_id}.csv"
            info_dir = f"{base_txt}_{new_id}.txt"
            print(f"Creating a new file: {data_dir} and {info_dir}")

    params = env_cfg.copy()
    if policy_type == "simple":
        trajectory = Simple_Trajectory(policy_name)
        trajectory.get_run_params(n_ep, n_seed , start_seed  , data_dir , info_dir , params)
        trajectory.get_policy_params(low_threshold, high_threshold)
    elif policy_type == "EP":
        trajectory = EP_Trajectory(policy_name)
        trajectory.get_run_params(n_ep, n_seed , start_seed  , data_dir , info_dir , params)
        trajectory.get_policy_params(low_threshold, high_threshold , price_offset , lookahead)
    elif policy_type == "Tsuction":
        trajectory = Tsuction_Trajectory(policy_name)
        trajectory.get_run_params(n_ep, n_seed , start_seed  , data_dir , info_dir , params)
        trajectory.get_policy_params(low_threshold, high_threshold , T_suction_setpoint)
    elif policy_type == "RL":
        trajectory = RL_Trajectory(policy_name)
        trajectory.get_run_params(n_ep, n_seed , start_seed  , data_dir , info_dir , params , hp_cfg.copy())
        trajectory.get_policy_params(model_dir , algo)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Supported types are simple, EP, Tsuction, RL")

    trajectory.set_policy()
    trajectory.get_trajectory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--C_path', type=str, default='env_config/C1.yaml', help="location of yaml config file to use for environment parameters, relative to ROOT dir of project")
    parser.add_argument('--HP_path', type=str, default='hyperparameters/H1.yaml', help="location of yaml config file to use for hyperparameters, relative to ROOT dir of project")
    parser.add_argument('--E_path', type=str, default='eval_config/E1.yaml', help="which policy or agent to use ")
    
    ARGS = parser.parse_args()
    print(ARGS)

    main()