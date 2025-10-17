
from typing import Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from stable_baselines3.common.utils import set_random_seed
from helpers import helper as h ### helper functions to make devices and rooms
from helpers import device as d ### device classes

class state_counter(gym.Wrapper): 
    '''
    Wrapper to count the number of times each state has been visited.
    Note: this is a very naive implementation and may not scale well with large state spaces.
    '''
    def __init__(self, env , key: str = "room_temperatures",low = -22.1, high = -15.9, bin = 62):
        super().__init__(env)
        self.N = int(bin)
        self.counter = np.zeros((self.N, self.N), dtype=np.int64)
        self.key = key
        self.room_vals = np.linspace(low, high, num=self.N, dtype=np.float64)
        self.V_MIN, self.V_MAX = float(self.room_vals[0]), float(self.room_vals[-1])
        self.DX = (self.V_MAX - self.V_MIN) / (self.N - 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._bump(obs)
        return obs, info
    
    def value_to_index(self, x: float) -> int:
        idx = int(round((x - self.V_MIN) / self.DX))
        return int(np.clip(idx, 0, self.N - 1))

    def snap2grid(self, x: float) -> float:
        return self.room_vals[self.value_to_index(x)]
    

    def _bump (self, obs ):
        room_temps = np.asarray(obs["room_temperatures"]).copy()
        i =  self.value_to_index(room_temps[0])
        j =  self.value_to_index(room_temps[1])
        self.counter[i,j]+=1


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._bump(obs)
        return obs, reward, terminated, truncated, info
    
    def get_counter(self) -> np.ndarray:
        return self.counter
    
    def save_npz(self, path: str):
        np.savez_compressed(path, grid=self.counter)

def make_env(params: dict, 
             render_mode: str = None, 
             rank: int = 0, 
             seed: int = 0,
             timelimit: int = 1440) -> Callable:
    """
    Factory function to create a new  Refrigeration environment instance. 
    Inputs
        params:      requires paramter set imported from yaml or manually
        render_mode: type of rendering to use, NotImplemented
        rank:        added to seed to allow for Vector Env
        seed:        seed for random number generator 
        timelimit:   set it in train file based on time steps and total time. 
    """
    def _init() -> gym.Env:
        env = Ref(params, render_mode)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=timelimit)
        env.reset(seed = seed + rank) ### seed the environment's RNG
        return env

    
    return _init

class Ref(gym.Env):

    """
    ### We did not implement any render modes yet, so this is just boilerplate code
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, 
                 params,
                 render_mode = None, 
                ):
        """
        NOTE: This environment is a simplified version of a refrigeration system with the following limitations:
         - Currently there is no discharge temperature
         - Currently there is no pressure control
         - We did not use CoolProp for this environment
         - The Suction Temperature is controlled directly, not through a PID controller (it is a setpoint for the compressor may be controllable or not)
         - The actions are on/off actions (continuous, fan VFD or binary) and Suction Temperature control if stated in the config file
         - The environment might forces on/off at certain conditions (config file!)
                - The percentage of time the evaporator is working as a defrost mode (but it is not really the defrost)
                - Force to be on for a window after turning on (or off)!
                - Force to be off (or on) if the temperature is close to the violation limit
        
        """

        self.params = params
        # LOAD EACH VARIABLE ANYWAY FOR CONVENIENCE
        
        self.window_size = 512  # The size of the PyGame window
        # params that scale the problem (We did not implement any visualization)

        self.delta_t = params["env"]["time_step"]  # Time step in seconds
        self.total_t = params["env"]["Total_time"]  # Total time in seconds
        self.timelimit = int(np.ceil(self.total_t / self.delta_t))  # Time limit in time steps

        self.action_type = params["env"]["Action_type"]  # Type of action space (discrete or continuous)
        self.action_spaces = params["env"]["Action_space"]  #The action spaces (If on/off or T_suction control or both or ...)
        self.action_mode = params["env"]["Action_mode"]  # "joint" or "separate" (joint: all actions together as one action (for discrete only), separate: each action as a separate value in the action vector)

        self.state_space = params["env"]["state_space"]  # State space representation (List of strings to set the observation state space)

        ### Set some parameters to make devices given the action space or state space
        if self.action_type == "discrete":
            if "suction_temperature" in self.action_spaces:
                self.sequencers_T_num = params["sequencers"]["T_suction_num"]
                self.sequencers_T_start = params["sequencers"]["T_suction_start"]
                seq_control = "discrete"
            else:
                seq_control = None
                self.sequencers_T_num = None
                self.sequencers_T_start = None
        else:
            if "suction_temperature" in self.action_spaces:
                seq_control = "continuous"
                self.sequencers_T_num = None
                self.sequencers_T_start = None
            else:
                seq_control = None
                self.sequencers_T_num = None
                self.sequencers_T_start = None

        #######################################################
        price_flag = None
        if "electricity_price" in self.state_space:
            price_flag = "vector"
        elif "electricity_price_list" in self.state_space:
            price_flag = "list"
        elif params["reward"]["electricity_price_data"] is None:
            price_flag = None

        ##### make devices and rooms! 1.room 2.evaporators 3.sequencers(vessels)  4.compressors 5.reward shaping

        self.rooms = h.make_room(params["rooms"])  # List of room objects
        self.evaporators = h.make_evaporator(params["evaporators"],  self.action_type, self.timelimit, self.delta_t)  # List of evaporator objects
        self.sequencers = h.make_sequencer(params["sequencers"], seq_control)  # List of sequencer objects
        self.compressors = h.make_compressor(params["compressors"],params["sequencers"]["seq_list"])  # List of compressor objects
        
        self.reward = d.reward_shaping(params["reward"],self.delta_t , price_flag)  # Reward shaping object
        
        # TODO: Render modes are not implemented, left boiler plate code inplace
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        
        # TODO: Check the range and observable parameters
        if "EP_remaining" in self.state_space:
            params["reward"]["horizon"] = len(self.reward.electricity_price_data)  ### set the horizon to the length of the electricity price data if we are using EP_remaining in the state space
        self.observation_space = h.get_observation_space(self.state_space, params)  # Get the observation space from the helper function

        self.action_space = h.get_action_space(self.action_spaces, self.action_type , self.action_mode,  params)  # Get the action space from the helper function


        self._obs_getters = {
                "room_temperatures":      lambda: np.asarray([room.get_obs_temp() for room in self.rooms], dtype=np.float64),           # shape (n_rooms,)
                "evaporator_utilization": lambda: np.asarray([evap.get_utilization() for evap in self.evaporators], dtype=np.float32),  # shape (n_evaporators,)
                "evaporator_on_time":    lambda: np.asarray([evap.on_period for evap in self.evaporators], dtype=np.int32),          # shape (n_evaporators,)
                "evaporator_off_time":   lambda: np.asarray([evap.off_period for evap in self.evaporators], dtype=np.int32),         # shape (n_evaporators,)
                "electricity_price_list": lambda: np.asarray(self.reward.price_array, dtype=np.float32),   # shape (price_num,)
                "EP_remaining":          lambda: np.asarray(self.reward.remain_step, dtype=np.int64),  # shape (price_num,)
                "electricity_price":      lambda: np.asarray(self.reward.get_obs_price(), dtype=np.float32),  # shape (lookahead_time(price_num),)
            }
        
        self._info_getters = {
            "time_step":               lambda: self.t,
            "Q_dist":                  lambda: np.asarray([room.Q_dist for room in self.rooms], dtype=np.float32),           # shape (n_rooms,)
            "compressor_power":        lambda: np.asarray([comp.W for comp in self.compressors], dtype=np.float32),      # shape (n_compressors,)
            "W_max":                   lambda: np.asarray([comp.W_max for comp in self.compressors], dtype=np.float32),  # shape (n_compressors,)
            "W_max_clip":              lambda: np.asarray([comp.W_max_clipped for comp in self.compressors], dtype=np.float32),  # shape (n_compressors,)
            "Q_max":                   lambda: np.asarray([comp.Q_max for comp in self.compressors], dtype=np.float32),  # shape (n_compressors,)
            "Q_max_clip":              lambda: np.asarray([comp.Q_max_clipped for comp in self.compressors], dtype=np.float32),  # shape (n_compressors,)
            "Q_evap":                  lambda: np.asarray([evap.get_Q_evap() for evap in self.evaporators], dtype=np.float32),  # shape (n_evaporators,)
            "overloaded":              lambda: np.asarray(self.is_overloaded , dtype=np.int32),  # shape (n_sequencers,)
            "T_suction":               lambda: np.asarray([seq.T_suction for seq in self.sequencers], dtype=np.float32),  # shape (n_sequencers,)
            "evaporator_status":       lambda: np.asarray([evap.on_off for evap in self.evaporators], dtype=np.float32),  # shape (n_evaporators,)
            "compressor_cost":         lambda: self.compressor_cost,  # Cost of running the compressor
            "violations_cost":         lambda: self.violations_cost,  # Cost of temperature violations
            "total_power":             lambda: self.W_total,  # Total power of the system


        }

    def _get_obs(self):
        '''
        Get the current observation of the environment.
        Returns:
        - obs: Current observation
        '''

        obs = {}
        for key in self.state_space:
            obs[key] = self._obs_getters[key]()
        return obs
    


    def _get_info(self):
        '''
        Get additional information about the environment.
        Returns:
        - info: Dictionary of additional information
        '''
        ### TODO: Add more information to the info dictionary if needed

        info = {key: self._info_getters[key]() for key in self._info_getters
        }
    

        return info

        
    def reset(self, seed=None, options=None , initial_state = None):
        """
        Initialize the environment using current parameters
        The electricite price and ambient temperature are the next time step values
        and the room temperatures are initialized to 0 farenheit.
        
        NOTE: options not used currently
        """
        super().reset(seed=seed)
        # seed action space
        self.action_space.seed(seed)
        
        self.t = 0
        
       ### TODO: Consider adding date and time to the environment
        #### Initialize variables (not tested yet!)
        if initial_state is not None:
            h.set_initial_state(self, initial_state , self.np_random, rooms = self.rooms , evaporators = self.evaporators , sequencers = self.sequencers , compressors = self.compressors)
        else:
            for room in self.rooms:
                room.room_reset(self.np_random)  # Reset room temperatures  # Placeholder for overload status
            for evap in self.evaporators:
                evap.evap_reset()  # Reset evaporator status
            for seq in self.sequencers:
                seq.seq_reset()  # Reset sequencer status
            for comp in self.compressors:
                comp.comp_reset()  # Reset compressor status
        self.reward.reward_reset()  # Reset reward shaping status
        self.compressor_cost = 0  # Cost of running the compressor
        self.violations_cost = 0  # Cost of temperature violations
        self.W_total = 0  # Total power of the system
        self.is_overloaded = [False] * len(self.sequencers)  # Placeholder for overload status

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    

    
    def step(self, action):

        #-------------------------------------------------------------------------------
        self.t += 1
        ###### Action
        
        ### The function to decode the input action based on the action space and type, 
        ### Currently it only supports setting evaporator on/off and suction temperature setpoint if in action space
        evap_actions , sequencer_actions = h.get_action (action, self.action_spaces, self.action_type , self.action_mode , self.evaporators , self.sequencers , self.sequencers_T_num , self.sequencers_T_start)
        

        if "suction_temperature" in self.action_spaces:
            for i , seq in enumerate(self.sequencers):
                if seq.check_time(self.t - 1):
                    seq.get_T_suction(sequencer_actions[i])

        #-------------------------------------------------------------------------------
        #### Compressor
        for comp in self.compressors:
            if comp.check_time(self.t - 1): # Befor any update, check if the device is sync with the environment time step
                comp.compressor_step(self.sequencers) ### Update compressor power based on the sequencer suction temperatures
        #-------------------------------------------------------------------------------

        #### Evaporator
        if "evaporator_control" in self.action_spaces and self.action_mode == "joint" and self.action_type == "discrete":

            ''' 
            0 is all off, 1 is first on, 2 is second on, ..., n is all on, Then, 1 is 0....01 but we want it to be 10...0 so we flip it
            For heuristic or rule based control, then we input sum (action[i] * 2**i) for evaoporator i

            '''
            evap_actions = np.flip(evap_actions)  # Flip action for on/off control only for joint discrete action space
            
        for i, evap in enumerate(self.evaporators):
            if evap.check_time(self.t - 1): # Befor any update, check if the device is sync with the environment time step
                if "evaporator_control" in self.action_spaces:
                    evap.evap_step(evap_actions[i], self.rooms, self.sequencers) ### Update evaporator status based on the action

        #----------------------------------------------------------------------------
        self.is_overloaded = h.set_all_Q_evap(self.evaporators, self.compressors ,self.sequencers)  # Set the evaporator heat loads based on the compressor capacities and suction temperatures
        #-------------------------------------------------------------------------------
        
        #### Room temperature update


        for i, room in enumerate(self.rooms):  # Update heat load of each room
            if room.check_time(self.t - 1): # Befor any update, check if the device is sync with the environment time step
                room.room_step(self.evaporators, self.delta_t , self.np_random)  # Update room temperature based on heat load and evaporator status
        #-------------------------------------------------------------------------------
        #### Compressor Power Update
        self.W_total = 0
        self.W_total = h.get_all_power(self.compressors , self.evaporators, self.sequencers)  # Get the total power consumption of all compressors
        
        #-------------------------------------------------------------------------------
        #### Electricity price update
        reward_value = 0
        # Initialize rewards
        
        if self.reward.check_time(self.t-1): # Befor any update, check if the device is sync with the environment time step
            self.compressor_cost , self.violations_cost , reward_value = self.reward.compute_reward( self.rooms , self.W_total)
            
        #-------------------------------------------------------------------------------

        terminated = False
        # Prepare observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()

        ##### Observation, reward, terminated, truncated, info (truncated is set by wappers of gym)

        return observation, reward_value, terminated, False, info




    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        raise NotImplementedError("TODO: No render modes implemented, set render_mode to None")

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
