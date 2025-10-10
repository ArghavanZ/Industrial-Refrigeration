import numpy as np
from typing import Callable
import gymnasium as gym
from gymnasium import spaces
from .device import sequencer,BaseCompressor,BaseEvaporator,ComplexEvaporator,room


def make_room (params : dict) -> list[room]:
    """
    Creates a list of room objects with the given parameters.
    """
    room_params = params.copy() ### copy the parameters to avoid modifying the original dictionary
    room_num = room_params["num_rooms"] ### get number of rooms
    rooms = [] # list to hold all room objects
    room_params.pop("num_rooms") 

    for i in range(room_num):
        room_param = {} # dictionary to hold parameters for each room
        for key in room_params:
            if room_params[key]  != "None": # check if the key is not "None", IF IS set to None.
                room_param[key] = room_params[key][i]
            else:
                room_param[key] = None
                
        
        rooms.append(room(params=room_param, id=i))

    return rooms



def make_evaporator(params : dict , action_type: str , timelimit , delta_t) -> list[BaseEvaporator | ComplexEvaporator]:
    """
    Creates a list of evaporator objects with the given parameters.
    """
    evaporator_params = params.copy() ### copy the parameters to avoid modifying the original dictionary
    evap_num = evaporator_params["num_evaps"] ### get number of evaporators
    evaporators = [] # list to hold all evaporator objects
    types = evaporator_params["types"]
    evaporator_params.pop("num_evaps")
    evaporator_params.pop("types")

    for i in range(evap_num):
        evaporator_param = {} # dictionary to hold parameters for each evaporator
        for key in evaporator_params:
            if evaporator_params[key] == "Y":
                evaporator_param[key] = True
            
            elif evaporator_params[key] != "None": # check if the key is not "None", IF IS set to None.
                evaporator_param[key] = evaporator_params[key][i]

            else:
                evaporator_param[key] = None
                
        if types[i] == "base":
            evaporators.append(BaseEvaporator(params=evaporator_param, id=i, action_type=action_type))
        elif types[i] == "complex":
            evaporators.append(ComplexEvaporator(params=evaporator_param, id=i, timelimit=timelimit , delta_t=delta_t, action_type=action_type))

    return evaporators



def make_sequencer(params: dict , action_type: str) -> list[sequencer]:
    """
    Creates a list of sequencer objects with the given parameters.
    """
    sequencer_params = params.copy() ### copy the parameters to avoid modifying the original dictionary
    sequencer_num = len(sequencer_params["seq_list"]) ### get number of sequencers
    sequencers = [] # list to hold all sequencer objects
    groups = sequencer_params["seq_list"]
    sequencer_params.pop("seq_list")

    for i in range(sequencer_num):
        sequencer_param = {} # dictionary to hold parameters for each sequencer
        for key in sequencer_params:
            if sequencer_params[key]  != "None": # check if the key is not "None", IF IS set to None.
                sequencer_param[key] = sequencer_params[key][i]
            else:
                sequencer_param[key] = None
        sequencers.append(sequencer(group=groups[i], id=i , params=sequencer_param , control = action_type))

    return sequencers




def make_compressor(params : dict, sequencers : list[sequencer]) -> list[BaseCompressor]:
    """
    Creates a list of compressor objects with the given parameters.
    Each compressor is associated with a sequencer.
    """
    compressor_params = params.copy() ### copy the parameters to avoid modifying the original dictionary
    compressor_num = compressor_params["num_compressors"] ### get number of compressors for each sequencer
    compressors = [] # list to hold all compressor objects
    id = 0

    for i , seq in enumerate(sequencers): ### for each sequencer create the number of compressors specified in compressor_num
        for j in range(compressor_num[i]): ### create compressors for each sequencer
            comp_param = {} # dictionary to hold parameters for each compressor
            for key in compressor_params[seq]:
                if compressor_params[seq][key] != "None": # check if the key is not "None", IF IS set to None.
                    comp_param[key] = compressor_params[seq][key][j]
                else:
                    comp_param[key] = None
            if comp_param["compressor_model"] == "base":
                comp_param.pop("compressor_model")
                compressors.append(BaseCompressor(params=comp_param, sequencer=seq , id=id))
                id += 1

    return compressors


def get_ambient_temperature(t , room_id):
    """
    Returns the ambient temperature at time t for room_id.
    This is a placeholder function and should be replaced with Data.
    decide if it should be in celsius or fahrenheit
    """
    # Example: Ambient temperature oscillates between 20 and 30 degrees Celsius
    return np.random.uniform(20, 30)  # the data or the parameters changes based on the input 


def celsius_to_fahrenheit(celsius):
    """
    Converts Celsius to Fahrenheit.
    """
    return celsius * 9.0 / 5.0 + 32.0

def fahrenheit_to_celsius(fahrenheit):
    """
    Converts Fahrenheit to Celsius.
    """
    return (fahrenheit - 32.0) * 5.0 / 9.0

def psig_to_pa(psig):
    """
    Converts pounds per square inch gauge (psig) to pascals (Pa).
    """
    return psig * 6894.76

def pa_to_psig(pa):
    """
    Converts pascals (Pa) to pounds per square inch gauge (psig).
    """
    return pa / 6894.76 

def get_Q_dis(mu, sigma, min_val, max_val):

    """
    Returns the heat load of each room.
    This is a placeholder function and should be replaced with actual data if it depends on time.
    The heat load can be influenced by various factors such as ambient temperature, room occupancy, and
    external heat sources.
    The function should return a value in watts (W)
    It supports having different Q_dist for each room (For example different mean and standard deviation for each room)
    """

    Q_dist  = np.random.normal(mu, sigma)  # Generate a random heat load for all the rooms based on the size of mu and sigma 
    Q_dist = np.clip(Q_dist, min_val, max_val)  # Ensure the heat load is within specified limits
    # Example: Q_dis oscillates between 10 and 20 kW
    return Q_dist


def get_electricity_price(t):
    """
    Returns the electricity price at time t.
    This is a placeholder function and should be replaced with actual data if it depends on time.
    The price can vary based on factors such as time of day, demand, and market conditions.
    The function should return a value in currency per kilowatt-hour (e.g., $/kWh) or watt-second (e.g., $/Ws).
    """
    # Example: Electricity price oscillates between 0.1 and 0.3 $/kWh
    return np.random.uniform(0.1, 0.3)

def set_all_Q_evap(evaporators, compressors , sequencers):
    """
    Calculate the evaporator heat transfer for each room.
    Q_evap = b_evap * (room_temps - suction_temp) * on_off_evap
    """
    groups = [x.group for x in sequencers]
    Q_max = []
    Q_evap = []
    for group in groups:
        Q_max.append(sum([comp.set_Q_max() for comp in compressors if comp.group == group]))

        Q_evap.append(sum([evap.compute_Q_evap() for evap in evaporators if evap.group == group]))
    scale_factors = [1] * len(groups)  ### initialize scale factors to 1
    is_scaled = [False] * len(groups)  ### initialize is_scaled to False for each group
    Q_total = np.minimum(np.array(Q_evap), np.array(Q_max))  ### ensure that Q_evap does not exceed Q_max
    for i, Q in enumerate(Q_evap):
        if Q > 0:
            scale_factors[i] = Q_total[i] / Q  ### compute scale factors for each group
        else :
            scale_factors[i] = 1
        scale_factors[i] = np.nan_to_num(scale_factors[i])  ### replace NaN with 0 (if Q_evap is 0, scale factor should be 0)
        is_scaled[i] = (scale_factors[i] != 1)  ### check if any scale factor is not 1


    for i , group in enumerate(groups):
        for evap in evaporators:
            if evap.group == group:
                evap.set_Q_evap(evap.compute_Q_evap() * scale_factors[i])  ### scale Q_evap for each evaporator in the group

    return is_scaled

def get_all_power(compressors , evaporators, sequencers):
    """
    Calculate the total power consumption of all compressors.
    """
    groups = [x.group for x in sequencers]
    Q_evap_total = [0] * len(groups)
    for i, group in enumerate(groups):
        Q_evap_total[i] = sum([evap.get_Q_evap() for evap in evaporators if evap.group == group])
        if Q_evap_total[i] < 0:
            Q_evap_total[i] = 0
        for comp in compressors:
            if comp.group == group:
                Q_evap = np.minimum(comp.Q_max , Q_evap_total[i])
                if Q_evap > 0:
                    comp.get_power(Q_evap)
                    Q_evap_total[i] -= Q_evap
                else:
                    comp.W = 0

    # print(f"Q_evap_total: {Q_evap_total} for groups: {groups} at t={evaporators[0].time_step}") ### for debugging
    # if np.any(np.array(Q_evap_total) != 0 ): 
    #     raise ValueError("Error in power calculation: Not all evaporator loads are met.")
    total_power = sum([comp.W for comp in compressors])
    return total_power

def compute_rewards( compressors, rooms, electricity_price , delta_t ,low , high):
    """
    Compute the reward for the current time step.
    The reward can be based on factors such as energy consumption, temperature regulation, and operational costs.
    This is a placeholder function and should be customized based on the specific objectives of the environment.
    """
    total_cost = 0
    compressor_cost = sum([comp.get_power() * electricity_price for comp in compressors]) * delta_t  # Total cost of all compressors
    violation_penalty = sum([room.get_violation_penalty(low[room.id] , high[room.id]) for room in rooms])  # Total penalty for all rooms
    total_cost -= compressor_cost + violation_penalty

    return compressor_cost, violation_penalty, total_cost


def get_observation_space(state_space , params)-> gym.spaces.Space:
    """
    Returns the observation space for the environment.
    The observation space can be a combination of different spaces such as Box, Discrete, MultiBinary, etc.
    This is a placeholder function and should be customized based on the specific state representation of the environment after adding new features.
    """
    observation_space = {}
    if "room_temperatures" in state_space:
        high = np.array(params["rooms"]["temperature_space_high"])
        low = np.array(params["rooms"]["temperature_space_low"])

        observation_space["room_temperatures"] = spaces.Box(low=low, high=high, dtype=np.float64)
       
        
    if "evaporator_utilization" in state_space:
        num_evaps = params["evaporators"]["num_evaps"]
        observation_space["evaporator_utilization"] = spaces.Box(low=0, high=100, shape=(num_evaps,), dtype=np.float32)

    if " evaporator_on_time" in state_space:
        num_evaps = params["evaporators"]["num_evaps"]
        delta_t = params["env"]["time_step"]  # Time step in seconds
        total_t = params["env"]["Total_time"]  # Total time in seconds
        timelimit = int(np.ceil(total_t / delta_t))  # Time limit in time steps
        observation_space["evaporator_on_time"] = spaces.Box(low=0, high=timelimit, shape=(num_evaps,), dtype=np.int32)

    if " evaporator_off_time" in state_space:
        num_evaps = params["evaporators"]["num_evaps"]
        delta_t = params["env"]["time_step"]  # Time step in seconds
        total_t = params["env"]["Total_time"]  # Total time in seconds
        timelimit = int(np.ceil(total_t / delta_t))  # Time limit in time steps
        observation_space["evaporator_off_time"] = spaces.Box(low=0, high=timelimit, shape=(num_evaps,), dtype=np.int32)

    if "electricity_price_list" in state_space:
        shape = params["reward"]["price_num"]
        min_price = params["reward"]["min_price"]
        max_price = params["reward"]["max_price"]
        observation_space["electricity_price_list"] = spaces.Box(low=min_price, high=max_price, shape=(shape,), dtype=np.float32)
    
    if "EP_remaining" in state_space: ### better to be box instead of discrete as it understand the order of the values
        shape = params["reward"]["price_num"]
        horizon = params["reward"]["horizon"]
        observation_space["EP_remaining"] = spaces.Box(low=0, high=horizon, shape=(shape,), dtype=np.int64)

    if "electricity_price" in state_space:
        min_price = params["reward"]["min_price"]
        max_price = params["reward"]["max_price"]
        shape = params["reward"]["price_num"]
        observation_space["electricity_price"] = spaces.Box(low=min_price, high=max_price, shape=(shape,), dtype=np.float32)


    return spaces.Dict(observation_space)
    

def get_action_space(Action_space ,Action_type, Action_mode, params) -> gym.spaces.Space:

    """
    Returns the action space for the environment.
    The action space can be a combination of different spaces such as Box, Discrete, MultiBinary, etc.
    This is a placeholder function and should be customized based on the specific action representation of the environment after adding new features.
    """
    if Action_mode == "separate" :
        if Action_type == "discrete":
            if "evaporator_control" in Action_space:
                if "suction_temperature" in Action_space:
                    num_evaps = params["evaporators"]["num_evaps"]
                    num_t_suction = len(params["sequencers"]["seq_list"]) ### number of sequencers, each sequencer has one suction temperature control
                    T_suction_start = np.array(params["sequencers"]["T_suction_start"]) ### start of discrete values for each sequencer (it is a list of size num_t_suction)
                    T_suction_num = np.array(params["sequencers"]["T_suction_num"]) ### number of discrete values for each sequencer (it is a list of size num_t_suction)
                    action_spaces = spaces.MultiDiscrete(
                        [2] * num_evaps + [T_suction_num[i] for i in range(num_t_suction)], dtype=np.int64 , start=[0] * num_evaps + [T_suction_start[i] for i in range(num_t_suction)]
                        )
                    return action_spaces
                else:
                    num_evaps = params["evaporators"]["num_evaps"]
                    action_spaces = spaces.MultiDiscrete([2] * num_evaps) ### or spaces.MultiBinary(num_evaps)
                    return action_spaces
        
        elif Action_type == "continuous":
            if "evaporator_control" in Action_space:
                if "suction_temperature" in Action_space:
                    num_evaps = params["evaporators"]["num_evaps"]
                    num_t_suction = len(params["sequencers"]["seq_list"])
                    action_spaces = spaces.Box(
                        low=np.array([0] * num_evaps + [-1] * num_t_suction),
                        high=np.array([1] * num_evaps + [1] * num_t_suction),
                        shape=(num_evaps + num_t_suction,),
                        dtype=np.float32,
                    )
                    return action_spaces
                else:
                    num_evaps = params["evaporators"]["num_evaps"]
                action_spaces = spaces.Box(low=0, high=1, shape=(num_evaps,), dtype=np.float32)
                return action_spaces
    elif Action_mode == "joint":
        if Action_type == "discrete":
            if "evaporator_control" in Action_space:
                if "suction_temperature" in Action_space:
                    num_evaps = params["evaporators"]["num_evaps"]
                    num_t_suction = len(params["sequencers"]["seq_list"])
                    T_suction_num = np.array(params["sequencers"]["T_suction_num"]) ### number of discrete values for each sequencer 
                    number_of_actions = 2 ** num_evaps * np.prod(T_suction_num) ### total number of actions
                    action_spaces = spaces.Discrete(number_of_actions)
                    return action_spaces
                else:
                    num_evaps = params["evaporators"]["num_evaps"]
                    number_of_actions = 2 ** num_evaps ### total number of actions
                    action_spaces = spaces.Discrete(number_of_actions)
                    return action_spaces
        
        elif Action_type == "continuous": #### not different from separate continuous
            if "evaporator_control" in Action_space:
                if "suction_temperature" in Action_space:
                    num_evaps = params["evaporators"]["num_evaps"]
                    num_t_suction = len(params["sequencers"]["seq_list"])
                    action_spaces = spaces.Box(
                        low=np.array([0] * num_evaps + [-1] * num_t_suction),
                        high=np.array([1] * num_evaps + [1] * num_t_suction),
                        shape=(num_evaps + num_t_suction,),
                        dtype=np.float32,
                    )
                    return action_spaces
                else:
                    num_evaps = params["evaporators"]["num_evaps"]
                action_spaces = spaces.Box(low=0, high=1, shape=(num_evaps,), dtype=np.float32)
                return action_spaces

def get_action (action , Action_spaces , Action_type , Action_mode , evaporators , sequencers , seq_T_num = None , seq_T_start = None) -> tuple:
    """
    Returns the action for the environment.
    The action can be a combination of different actions .
    """
    evap_actions = None
    suction_actions = None
    if Action_mode == "separate" :
        if "evaporator_control" in Action_spaces:
            evap_actions = action[:len(evaporators)]
        if "suction_temperature" in Action_spaces:
            suction_actions = action[-len(sequencers):]
            
    elif Action_mode == "joint":
        if Action_type == "discrete":
            if "evaporator_control" in Action_spaces:
                if "suction_temperature" in Action_spaces:
                    num_evaps = len(evaporators)
                    num_t_suction = len(sequencers)
                    T_suction_num = np.array(seq_T_num) ### number of discrete values for each sequencer
                    T_suction_start = np.array(seq_T_start) ### starting values for each sequencer
                    evap_actions = [0] * num_evaps
                    suction_actions = [0] * num_t_suction
                    evap_act = action % (2 ** num_evaps) 
                    for i in range(num_evaps - 1, -1, -1):
                        evap_act, remainder = divmod(evap_act, 2)
                        evap_actions[i] = remainder
                    suction_act = action // (2 ** num_evaps)
                    for i in range(num_t_suction - 1, -1, -1):
                        suction_act, remainder = divmod(suction_act, T_suction_num[i])
                        suction_actions[i] = remainder + T_suction_start[i]
                else:
                    num_evaps = len(evaporators)
                    evap_actions = [0] * num_evaps
                    evap_act = action % (2 ** num_evaps) 
                    for i in range(num_evaps - 1, -1, -1):
                        evap_act, remainder = divmod(evap_act, 2)
                        evap_actions[i] = remainder
                    suction_actions = None
        elif Action_type == "continuous": #### not different from separate continuous
            if "evaporator_control" in Action_spaces:
                evap_actions = action[:len(evaporators)]
            if "suction_temperature" in Action_spaces:
                suction_actions = action[-len(sequencers):]
                
    return evap_actions , suction_actions

def set_initial_state(initial_state ,np_random, rooms , evaporators , compressors , sequencers):
    """
    Sets the initial state of the environment.
    The initial state can be a combination of different states such as room temperatures, evaporator states, compressor states, etc.
    This is a placeholder function and should be customized based on the specific state representation of the environment.
    """

    ## TODO: Check if the initialization support all possible evap on_off status (for complex evaporators)
    for room in rooms:
        room.reset_room_temp(np_random)  # Reset room temperatures  # Placeholder for overload status
    for evap in evaporators:
        evap.evap_reset()  # Reset evaporator status
    for seq in sequencers:
        seq.seq_reset()  # Reset sequencer status
    for comp in compressors:
                comp.comp_reset()  # Reset compressor status
    if "room_temperatures" in initial_state:
        for i, room in enumerate(rooms):
            room.temp = initial_state["room_temperatures"][i]
    
    if "evaporator_status" in initial_state:
        for i, evap in enumerate(evaporators):
            evap.on_off = initial_state["evaporator_status"][i]


    if "evaporator_on_time" in initial_state:
        for i, evap in enumerate(evaporators):
            evap.on_period = initial_state["evaporator_on_time"][i]
            evap.last_action = evap.on_off  # Ensure last_action matches the initial on/off status
            evap.on_off_total = int(evap.on_off > 0)
    if "evaporator_off_time" in initial_state:
        for i, evap in enumerate(evaporators):
            evap.off_period = initial_state["evaporator_off_time"][i]
            evap.last_action = evap.on_off  # Ensure last_action matches the initial on/off status
            evap.on_off_total = int(evap.on_off > 0)

    if "evaporator_utilization" in initial_state:
        for i, evap in enumerate(evaporators):
            evap.util = initial_state["evaporator_utilization"][i]
            if evap.util_window is not None:
                if initial_state["evaporator_on_off_window"][i] == evap.util_window:
                    evap.on_off_window = initial_state["evaporator_on_off_window"][i]
                    evap.on_off_total = sum(int(x)>0 for x in evap.on_off_window)
                else:
                    raise ValueError(f"Initial state error: Evaporator {i} utilization window and on/off window do not match.")
            
    if "compressor_suction_temp" in initial_state:
        for i, comp in enumerate(compressors):
            comp.T_suction = initial_state["compressor_suction_temp"][i]

    if "sequencer_suction_temp" in initial_state:
        for i, seq in enumerate(sequencers):
            seq.T_suction = initial_state["sequencer_suction_temp"][i]

    return