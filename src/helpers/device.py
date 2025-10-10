import numpy as np



class room:
    def __init__(self, params , id):
        self.params = params
        self.id = id

        #### Q_dist parameters for each room
        self.mu_dist = params["mu_dist"] # list of float, mean heat load for each room (in W)
        self.sigma_dist = params["sigma_dist"] # list of float, standard deviation of heat load for each room (in W)

        self.min_dist = params["min_dist"] # list of float, minimum heat load for each room (in W)
        self.max_dist = params["max_dist"] # list of float, maximum heat load for each room (in W)

        ##### Room temperature parameters
        self.min_temp = params["room_min_temp"] # float, minimum acceptable temperature for each room (in degree Celsius)
        self.max_temp = params["room_max_temp"] # float, maximum acceptable temperature for each room (in degree Celsius)

        self.delta_dist = params["delta_dist"] # list of float, change in heat load for each room (in W)
        self.dist_dist = params["dist_dist"] # list of float, time interval for change in heat load for each room (in number of time steps)

        ##### observation space
        self.high_value = params["temperature_space_high"]
        self.low_value = params["temperature_space_low"]

        #### Heat capacity 
        self.c_room = params["c_room"] # float, thermal capacity of each room

        self.init_temp_low = params["room_init_temp_low"] # float, initial temperature range of each room (in degree Celsius)
        self.init_temp_high = params["room_init_temp_high"] # float, initial temperature range of each room (in degree Celsius)

        self.c_ambient = params["c_ambient"] # float, thermal capacity of the ambient environment
        self.ambient_temp = params["Ambient_temperature"] # float, initial temperature of the ambient environment (in degree Celsius)

        self.evap_list =  params["evap_list_id"] ### list of evaporator in each room 

        self.T_room_sp = (self.min_temp + self.max_temp) / 2.0  ### setpoint for base policy! 

        self.time_step = 0

        self.Q_dist = 0

    def room_reset(self, np_random): ### set initial temp 
        self.time_step = 0
        self.temp = np_random.uniform(self.init_temp_low, self.init_temp_high)
        return self.temp
    
    def room_step(self, evaporators, delta_t , np_random):
        """
        Performs a single time step update for the room.
        """
        T_ambient = self.get_ambient_temp(self.time_step)  # Placeholder for ambient temperature function
        Q_evap = self.get_Q_evap(evaporators)
        Q_dist = self.get_Q_dist(np_random)
        self.update_room_temp(Q_evap, Q_dist, T_ambient, delta_t)
        self.time_step += 1

    def check_time(self , t):
        return self.time_step == t


    def get_obs_temp(self): ### each time we get the room temp for observation, we clip it but we do not change it inside the environment
        return np.clip(self.temp, self.low_value, self.high_value)

    def get_room_temp(self): ### return true temp 
        return self.temp

    def get_Q_dist(self,np_random):
        """
        Returns the heat load of each room.(It set the Q_dist attribute of the room class)
        This is a placeholder function and should be replaced with actual data if it depends on time.
        The heat load can be influenced by various factors such as ambient temperature, room occupancy, and
        external heat sources.
        The function should return a value in watts (W)
        It supports having different Q_dist for each room (For example different mean and standard deviation for each room)
        """

        if self.dist_dist == "normal":
            self.Q_dist = np_random.normal(self.mu_dist, self.sigma_dist)  # Generate a random heat load for all the rooms based on the size of mu and sigma
            self.Q_dist = np.clip(self.Q_dist, self.min_dist, self.max_dist)  # Ensure the heat load is within specified limits
        elif self.dist_dist == "uniform":
            self.Q_dist = np_random.uniform(self.mu_dist - self.delta_dist, self.mu_dist + self.delta_dist)  # Generate a random heat load for all the rooms based on the size of mu and sigma
        else:
            raise ValueError("Invalid distribution type. Please choose 'normal' or 'uniform'.")
        # Example: Q_dis oscillates between 10 and 20 kW
        return self.Q_dist
    

    def get_room_group(self, evaporators): ### what group the room is in
        """
        Returns the group of rooms to which this room belongs.
        """
        evaporator_groups = []
        for i in self.evap_list:
            evaporator_groups.append(evaporators[i].group)

        if all(x == evaporator_groups[0] for x in evaporator_groups): ### if all the evaporators in the room are in the same group
            return evaporator_groups[0]
        else:
            return None

    def get_evap_list(self):  ### get the int list of evaporators
        evap_list = [int(i) for i in self.evap_list]
        return evap_list

    def get_Q_evap(self, evaporators):
        """
        Returns the total heat removed by all evaporators in the room (in watts).
        """
        Q_evap = sum(evaporators[i].get_Q_evap() for i in self.evap_list)
        return Q_evap

    
    def update_room_temp(self, Q_evap, Q_dist, T_ambient, delta_t):
        """
        Updates the room temperature based on the heat removed by the evaporator and the heat load.
        Q_evap: Total heat removed by all evaporators in the room (in watts)
        Q_dist: Heat load of the room (in watts)
        T_ambient: Ambient temperature (in degree Celsius)[the heat exchange with ambient]
        delta_t: Time step (in seconds)
        """
        
        if self.c_ambient is None:
            dT = (Q_dist - Q_evap) * delta_t / self.c_room
            

        else: 
            dT = (Q_dist - Q_evap + self.c_ambient * (T_ambient - self.temp)) * delta_t / self.c_room

        # Update the room temperature (we are not really clipping the room temperature)
        self.temp += dT 
        

    def get_ambient_temp(self, t):
        """
        Returns the ambient temperature of the room.
        later replace with actual data grabber
        """

        return self.ambient_temp

    def get_room_constraint(self):
        """
        Returns the room temperature constraints (min and max).
        """
        return self.min_temp, self.max_temp


class BaseEvaporator:
    def __init__(self, params, id , action_type = "discrete"):
        self.params = params
        self.id = id
        self.room = params["room_ids"]  # Room ID to which the evaporator belongs
        self.b_evap = params["b_evap"] # float, evaporator capacity (in watt per degree Celsius)
        self.group = params["sequencer"]
        self.time_step = 0
        self.on_off = 0 ### is off
        self.continuous = action_type == "continuous"
        self.Q_evap = 0 ### initial Q_evap
        

    def evap_reset(self):
        self.time_step = 0
        self.on_off = 0 ### is off
        self.Q_evap = 0 ### initial Q_evap


    def evap_step(self , action , rooms, sequencers):
        """
        Performs a single time step update for the evaporator.
        """
        self.T_room = rooms[self.room].get_room_temp() ### first get the room temp
        self.set_on_off(action)
        # self.T_suction = self.get_suction_temp(t)  ### if we want to have time varying suction temp
        self.T_suction = self.set_T_suction(sequencers)
        self.time_step += 1


    
    def check_time(self , t):
       return self.time_step == t

    def get_on_off(self):
        return self.on_off

    
    def set_on_off(self, action , min_temp = None , max_temp = None):

        if self.continuous:
            self.on_off = action
        else: # from binary to integer
            self.on_off = int(action)


    def set_T_suction(self, sequencers):
        """
        Set the suction temperature for the evaporator based on the sequencer parameters.
        """
        seq_id = next(i for i, seq in enumerate(sequencers) if seq.group == self.group)
        self.T_suction = sequencers[seq_id].T_suction 
        return self.T_suction 

    def compute_Q_evap(self):
        """
        Calculate the evaporator heat transfer for each room.
        Q_evap = b_evap * (room_temps - suction_temp) * on_off_evap
        Remember Q_evap will be scaled considering Q_max later
        """
        Q_evap = self.b_evap * (self.T_room- self.T_suction) * self.on_off
        Q_evap = np.maximum(Q_evap, 0)  # Ensure the heat removal is non-negative
        return Q_evap

    def get_Q_evap(self):
        return self.Q_evap
    
    def set_Q_evap(self, Q_evap):
        self.Q_evap = Q_evap


class ComplexEvaporator(BaseEvaporator):
    def __init__(self, params, id , timelimit , delta_t , action_type = "discrete"):
        super().__init__(params, id, action_type)
        self.on_period = 0 ### how long it has been on back to back
        self.off_period = 0 ### how long it has been off back to back
        self.last_action = 0 ### what was the last action (on/off)
        self.on_off_total = 0 
        self.util = 0 
        self.delta_t = delta_t ### time step in seconds
        ### Complex evaporator parameters

        # on/off time constraints
        self.min_on_time = params["min_on_time"] ### minimum allowed back to back on time
        self.min_on_time = int(np.ceil(self.min_on_time / self.delta_t))
        self.min_off_time = params["min_off_time"] ### minimum allowed back to back off time
        self.min_off_time = int(np.ceil(self.min_off_time / self.delta_t))
        self.on_off_min_constraint = params["on_off_min_constraint"] ### if None, don't apply on/off min time constraints

        # on/off buffer constraints
        self.on_off_buffer = params["on_off_buffer"] ### if None, don't apply on/off buffer constraints (Y with 0 is different from None, None means no buffer constraints)
        self.low_buffer = params["temp_buffer_low"] ### low room temp buffer
        self.high_buffer = params["temp_buffer_high"] ### high room temp buffer

        # on/off percentage constraints
        self.util_constraint = params["util_constraint"] ### if None, don't apply utilization percentage constraints
        self.util_per = params["util_per"] ### utilization percentage constraint (how much percentage of the time the evaporator can be on)
        self.util_window = params["util_window"] ### utilization window (in number of time steps, if None, we consider the whole episode)
        if self.util_window != 'None': ### if we have a window, we need to keep track of the on/off history for a window
            self.on_off_window = np.zeros(self.util_window)
        self.timelimit = timelimit ### total time limit of the episode in number of time steps if we do not have a window

        
    def evap_reset(self):
        super().evap_reset()
        self.on_period = 0
        self.off_period = 0
        self.last_action = 0
        self.on_off_total = 0
        self.util = 0
        if self.util_window != None:
            self.on_off_window = np.zeros(self.util_window) 

    def evap_step(self, action, rooms, sequencers):
        """
        Performs a single time step update for the evaporator.
        """
        self.T_room = rooms[self.room].get_room_temp() ### first get the room temp
        min_temp, max_temp = rooms[self.room].get_room_constraint()
        self.set_on_off(action , min_temp , max_temp)
        # self.T_suction = self.get_suction_temp(t)  ### if we want to have time varying suction temp
        self.T_suction = self.set_T_suction(sequencers)
        self.time_step += 1
        


    def set_on_off(self, action , min_temp , max_temp):
        """
        Set the on/off state of the evaporator with constraints.
        """
        self.last_action = self.on_off ### set the last action
        super().set_on_off(action) ### set the current action
        ##### change them based on priority! 
        if self.on_off_min_constraint != None:
            self.apply_on_off_min_constraint()
        if self.on_off_buffer != None:
            self.apply_on_off_buffer_constraint(min_temp , max_temp)
        if self.util_constraint != None:
            self.apply_util_constraint()

        if self.continuous:
            self.on_off_total += int(self.on_off > 0) ### update the total on time
            if self.on_off > 0:
                self.on_period += 1 ### increase the on period
                self.off_period = 0 ### reset the off period
            else:
                self.off_period += 1 ### increase the off period
                self.on_period = 0 ### reset the on period
        else:
            self.on_off_total += self.on_off ### update the total on time
            if self.on_off == 1:
                self.on_period += 1 ### increase the on period
                self.off_period = 0 ### reset the off period
            else:
                self.off_period += 1 ### increase the off period
                self.on_period = 0 ### reset the on period

    
    def apply_on_off_min_constraint(self):
        """
        Apply the minimum on/off time constraint.
        """
        if self.continuous:
            if self.on_off == 1: # new action is on
                if self.last_action == 1: # last action is also on
                    self.on_period += 1 ### increase the on period
                    self.off_period = 0 ### reset the off period
                else: # last action is off
                    if self.off_period < self.min_off_time: # if the off period is less than the min off time, keep it off
                        self.on_off = 0 ### keep it off
            else: # new action is off
                if self.last_action == 0: # last action is also off
                    self.off_period += 1 ### increase the off period
                    self.on_period = 0 ### reset the on period
                else: # last action is on
                    if self.on_period < self.min_on_time: # if the on period is less than the min on time, keep it on
                        self.on_off = 1 ### keep it on

        else:
            if self.on_off > 0: # new action is on
                if self.last_action > 0: # last action is also on
                    self.on_period += 1 ### increase the on period
                    self.off_period = 0 ### reset the off period
                else: # last action is off
                    if self.off_period < self.min_off_time: # if the off period is less than the min off time, keep it off
                        self.on_off = 0 ### keep it off
            else: # new action is off
                if self.last_action == 0: # last action is also off
                    self.off_period += 1 ### increase the off period
                    self.on_period = 0 ### reset the on period
                else: # last action is on
                    if self.on_period < self.min_on_time: # if the on period is less than the min on time, keep it on
                        self.on_off = self.last_action ### keep it on

    def apply_on_off_buffer_constraint(self, min_temp , max_temp):
        """
        Apply the temperature buffer constraint.
        """
        if self.T_room - min_temp < self.low_buffer: # if the room temp is less than the low buffer, turn it off
            self.on_off = 0 ### turn it off
        elif max_temp - self.T_room < self.high_buffer: # if the room temp is more than the high buffer, turn it on
            self.on_off = 1 ### turn it on

    def apply_util_constraint(self):
        """
        Apply the utilization percentage constraint.
        """
        if self.util_window == None: 
            if self.continuous:
                if self.on_off > 0:
                    if self.util/self.timelimit > self.util_per/100:
                        self.on_off = 0 ### turn it off
                self.util += int(self.on_off > 0)
            else:
                if self.on_off == 1:
                    if self.util/self.timelimit > self.util_per/100:
                        self.on_off = 0 ### turn it off
                self.util += self.on_off
        
        else: 
            if self.time_step >= self.util_window:
                if self.continuous:
                    if self.on_off > 0:
                        if np.mean(self.on_off_window) > self.util_per/100:
                            self.on_off = 0
                            self.on_off_window = np.roll(self.on_off_window, -1)
                            self.on_off_window[-1] = 0
                        else:
                            self.on_off_window = np.roll(self.on_off_window, -1)
                            self.on_off_window[-1] = 1
                    else:
                        self.on_off_window = np.roll(self.on_off_window, -1)
                        self.on_off_window[-1] = self.on_off
                else:
                    if self.on_off == 1:
                        if np.mean(self.on_off_window) > self.util_per/100:
                            self.on_off = 0
                            self.on_off_window = np.roll(self.on_off_window, -1)
                            self.on_off_window[-1] = 0
                        else:
                            self.on_off_window = np.roll(self.on_off_window, -1)
                            self.on_off_window[-1] = 1
                    else:
                        self.on_off_window = np.roll(self.on_off_window, -1)
                        self.on_off_window[-1] = self.on_off
            else:
                if self.continuous:
                    if self.on_off > 0:
                        self.on_off_window[self.time_step] = 1
                    else:
                        self.on_off_window[self.time_step] = self.on_off
                else:
                    self.on_off_window[self.time_step] = self.on_off

    def get_utilization (self):

        if self.util_window == None:
            return self.util / self.timelimit * 100
        else:
            return np.mean(self.on_off_window) * 100
    
class sequencer:
    def __init__(self, group , id , params , control = None):
        self.group = group ### type of sequencer
        self.id = id ### id of sequencer
        self.control = control
        self.time_step = 0

        if control == None:
            self.T_suction_sp = params["T_suction_sp"] ### if no control is specified, we keep it constant
            self.T_suction = self.T_suction_sp #### Constant suction temp
        elif control == "continuous":
            self.T_suction_base = params["T_suction_sp"] # This is the base setpoint for the suction temperature, as a base setpoint to choose the suction temperature for the compressor.   --- IGNORE ---
            self.T_suction_scale = params["T_suction_scale"] # (in degree Celsius) This is the scale for the suction temperature action (T_suction = setpoint + scale * action, action in [-1, 1]) --- IGNORE ---
            self.T_suction = self.T_suction_base ### initial suction temp
        else:
            self.T_suction_start = params["T_suction_start"] #### discrete suction temp
            self.T_suction = self.T_suction_start ### initial suction temp

    def check_time(self , t):
        return self.time_step == t
    
    def seq_reset(self):
        self.time_step = 0
        if self.control == None:
            self.T_suction = self.T_suction_sp #### Constant suction temp
        elif self.control == "continuous":
            self.T_suction = self.T_suction_base ### initial suction temp
        else:
            self.T_suction = self.T_suction_start ### initial suction temp

        

    def get_T_suction(self , action):
        """
        Returns the suction temperature for the sequencer.
        The function should return a value in degree Celsius
        """

        if self.control == "continuous":
            self.T_suction = self.T_suction_base + self.T_suction_scale * action

        elif self.control == "discrete":
            self.T_suction = action

        return self.T_suction

    def sequencer_step(self,action):
        """
        Performs a single time step update for the sequencer.
        """
        self.get_T_suction(action)
        self.time_step += 1

class BaseCompressor:
    def __init__(self, params , sequencer, id):
        self.params = params
        self.group = sequencer
        self.id = id

        ### load parameters
        self.c_c = params["c_c_coeff"] # float, cooling capacity of the compressor (in watt per degree Celsius)
        self.Q_rated = params["Q_rated"] # Compressor rated heat removal at the evaporators (in Watt)
        self.T_s_rated = params["T_suction_rated"] # Compressor rated suction temperature (in degree Celsius)
        

        #### Compressor capacity clipping values
        self.Q_min_clip = params["Q_min_clip"] # Minimum compressor capacity (in Watt)
        self.Q_max_clip = params["Q_max_clip"]  # Maximum compressor capacity (in Watt)
        
        ##### Compressor power parameters
        ###  W_rated =  -0.2*T_suction_rated^2 + 0.2*T_suction_rated + alpha
        ###  W_max =  -0.2*T_suction^2 + 0.2*T_suction + alpha
        self.c_w1 = params["c_w_coeff1"] # (in Watt per degree Celsius squared)
        self.c_w2 = params["c_w_coeff2"] # (in Watt per degree Celsius)
        
        self.W_rated = params["W_rated"] # (in Watt)


        self.W_min_clip = params["W_min_clip"] # Minimum compressor power (in Watt)

        self.T_suction = self.T_s_rated ### initial suction temp

        ### Compressor power-capacity model 
        #### (W / W_max) = c_coeff1 * (Q_evap_total / Q_max) + c_coeff2 * (on/off))
        self.c_coeff1 = params["c_coeff1"] # 
        self.c_coeff2 = params["c_coeff2"] # (in watt)
        

        self.T_discharge = params["T_discharge"] # Discharge temperature of the evaporator (assumed constant and not implemented yet)
        self.time_step = 0

        self.W = 0
        self.W_max = self.W_rated
        self.Q_max = self.Q_rated
        self.W_max_clipped = 0
        self.Q_max_clipped = 0

    def check_time(self , t):
        return self.time_step == t
    
    def comp_reset(self):
        self.time_step = 0
        self.T_suction = self.T_s_rated ### initial suction temp
        self.W = 0
        self.W_max = self.W_rated
        self.Q_max = self.Q_rated
        self.W_max_clipped = 0
        self.Q_max_clipped = 0

    def compressor_step(self , sequencers):
        """
        Performs a single time step update for the compressor.
        """
        self.set_T_suction(sequencers)
        self.set_Q_max()
        self.compute_W_max()
        self.time_step += 1

    def set_T_suction(self,sequencers):
        """
        Set the suction temperature for the compressor based on the sequencer parameters.
        """
        seq_id = next(i for i, seq in enumerate(sequencers) if seq.group == self.group)
        self.T_suction = sequencers[seq_id].T_suction
        return self.T_suction


    def set_Q_max(self):
        """
        Returns the maximum cooling capacity of the compressor.
        """

        self.Q_max = self.Q_rated - self.c_c * (self.T_s_rated - self.T_suction)
        self.Q_max_clipped = 1*((self.Q_max < self.Q_min_clip)|(self.Q_max > self.Q_max_clip))
        self.Q_max = np.maximum(np.minimum(self.Q_max, self.Q_max_clip), self.Q_min_clip)  # Maximum compressor capacity at the current suction temperature (in watt), clipped to the minimum and maximum values

        return self.Q_max
    
    def compute_W_max(self):
        """
        Returns the maximum power consumption of the compressor.
        """

        self.W_max = (self.T_suction - self.T_s_rated)**2 * self.c_w1 + (self.T_suction - self.T_s_rated) * self.c_w2 + self.W_rated  # Compressor power at the current suction temperature (in watt)
        self.W_max_clipped = 1 * (self.W_max < self.W_min_clip)
        self.W_max = np.maximum(self.W_max, self.W_min_clip)
        return self.W_max
    


    def get_power(self, Q_evap):
        """
        Returns the power consumption of the compressor based on the evaporator heat removal.
        The function should return a value in watts (W)
        """
        if Q_evap > self.Q_max:
            raise ValueError("Evaporator heat removal exceeds compressor maximum capacity.")
        elif Q_evap < 0:
            raise ValueError("Evaporator heat removal cannot be negative.")
        else:
            self.W = (self.c_coeff1 * (np.minimum(Q_evap / self.Q_max , 1)) + self.c_coeff2 * (Q_evap>0)) * self.W_max  # Compressor power in watts
            return self.W


class reward_shaping:
    def __init__(self, params , delta_t , price_flag = None):
        self.params = params
        self.price_flag = price_flag
        self.reward_type = params["reward_type"] ### type of reward function
        self.temp_penalty_low = np.array(params["temp_penalty"]["low"])  # Weight for low temperature violation in the reward function
        self.temp_penalty_high = np.array(params["temp_penalty"]["high"])  # Weight for high temperature violation cost in the reward function
        self.electricity_cost = params["electricity_price"] # Weight for compressor cost in the reward function
        self.scale = params["reward_scale"] # reward scaling factor
        self.delta_t = delta_t
        if params["electricity_price_data"] == "None": ## Constant electricity price
            self.electricity_price_data = None
            self.price_array = None
            self.steps = None
            self.p_window = None ### wimdow of lookahead price data
        elif isinstance(params["electricity_price_data"], str) and price_flag != None: #### if it is a string of a file path
            if price_flag == "list":
                path = params["electricity_price_data"]
                self.electricity_price_data , self.price_array , self.steps = self.load_price( path) ### a numpy array of electricity price data
                self.e_time = 0  ### initial time step for electricity price data [reset when we finish the data file]
                self.remain_step = self.steps.copy() ### remaining steps in the data file
                self.p_window = None ### window of lookahead price data
            elif price_flag == "vector":
                path = params["electricity_price_data"]
                data = self.load_price( path) ### a numpy array of electricity price data
                self.electricity_price_data = data["price"] ### a numpy array of electricity price data
                self.e_time = 0  ### initial time step for electricity price data [reset when we finish the data file]
                self.steps = len(self.electricity_price_data)
                self.p_window = params["price_num"] ### window of lookahead price data
        elif isinstance(params["electricity_price_data"], list): ### if it is a list of values , periodicly repeat these values over an episode 
            self.electricity_price_data = np.array(params["electricity_price_data"])
            self.e_time = 0 ### initial time step for electricity price data [reset when we finish the list, one episode for now]
            self.steps = len(self.electricity_price_data)
            self.p_window = params["price_num"] ### window of lookahead price data
        else:
            raise ValueError("Invalid electricity_price_data format. Must be 'None', file path string, or list of values.")
        self.time_step = 0

    def check_time(self , t):
        return self.time_step == t
    
    def reward_reset(self):
        self.time_step = 0
    
    def get_electricity_price(self): #### to get the current step electricity price
        if self.electricity_price_data is None:
            return self.electricity_cost
        else:
            price = self.electricity_price_data[self.e_time]
            self.e_time += 1
            if self.price_flag == "list":
                nz = np.flatnonzero(self.remain_step > 0)
                if len(nz) > 0:
                    self.remain_step[nz[0]] -= 1
                    if self.e_time >= len(self.electricity_price_data):
                        self.e_time = 0 ### reset the time step if we finish the data file or list
                        nz_flag = np.flatnonzero(self.remain_step > 0)
                        if len(nz_flag) > 0:
                            raise ValueError("The electricity price data length is not consistent.")
                        self.remain_step = self.steps.copy() ### reset the remaining steps if we finish the data file
                self.electricity_cost = price
                return self.electricity_cost
            elif self.price_flag == "vector":
                if self.e_time >= len(self.electricity_price_data):
                    self.e_time = 0 ### reset the time step if we finish the data file or list
                self.electricity_cost = price
                return self.electricity_cost

    def load_price(self, path):
        """
        Load electricity price data from a file.
        The file should contain a list of electricity prices (in $/kWh) for each time step.
        """
        file_extension = path.split('.')[-1]
        if file_extension not in ['txt', 'csv' , 'npz' , 'npy']:
            raise ValueError("Unsupported file format. Please provide a .txt, .csv, .npz, or .npy file.")
        else:
            if file_extension in ['npz' , 'npy']: ### npy or npz
                try:
                    price_data = np.load(path)
                    if isinstance(price_data, np.lib.npyio.NpzFile):
                        price = price_data["price"]  # Load the first array in the .npz file
                        price_array = price_data["price_array"]
                        steps = price_data["steps"]

                    return price , price_array , steps
                except Exception as e:
                    raise ValueError(f"Error loading electricity price data from {path}: {e}")
        
            else: ### txt or csv (Is CSV load with np.loadtxt safe?)
                try:
                    price_data = np.loadtxt(path, delimiter=',')
                    return price_data
                except Exception as e:
                    raise ValueError(f"Error loading electricity price data from {path}: {e}")
                
    def get_obs_price(self): ### Mostly to get a window of price data for the observation [can be used when we do not have price_list in observation but only price !]
        if self.electricity_price_data is None:
            return self.electricity_cost
        else:
            if self.p_window == None :
                return self.electricity_cost
                
            else:
                if self.e_time + self.p_window <= len(self.electricity_price_data):
                    return self.electricity_price_data[self.e_time:self.e_time + self.p_window]
                else:
                    end_part = self.electricity_price_data[self.e_time:]
                    start_part = self.electricity_price_data[:self.p_window - len(end_part)]
                    return np.concatenate((end_part, start_part))
                
    def compute_reward(self, rooms , power ):
        
        """
        Compute the reward for the current time step.
        The reward can be based on factors such as energy consumption, temperature regulation, and operational costs.
        This is a placeholder function and should be customized based on the specific objectives of the environment.
        """
        self.time_step += 1
        self.get_electricity_price()
        total_cost = 0
        compressor_cost = power * self.electricity_cost   # Total cost of all compressors
        room_constraints = np.asarray([room.get_room_constraint() for room in rooms]).copy()  # Shape (num_rooms, 2)
        room_temps = np.asarray([room.get_room_temp() for room in rooms]).copy()
        low_room_violations =  np.maximum(room_constraints[:, 0] - room_temps, 0)
        high_room_violations = np.maximum(room_temps - room_constraints[:, 1], 0)  # Temperature violations high and low
        # room_violations = np.stack([low_room_violations, high_room_violations], axis=1)  # Shape (num_rooms, 2)
        if self.reward_type == "linear":
            violation_penalty = np.sum(low_room_violations * self.temp_penalty_low + high_room_violations * self.temp_penalty_high)  # Total penalty for all rooms
        elif self.reward_type == "quadratic":
            violation_penalty = np.sum((low_room_violations**2) * self.temp_penalty_low + (high_room_violations**2) * self.temp_penalty_high)  # Total penalty for all rooms
        elif self.reward_type == "exponential":
            violation_penalty = np.sum((np.exp(low_room_violations) - 1) * self.temp_penalty_low + (np.exp(high_room_violations) - 1) * self.temp_penalty_high)  # Total penalty for all rooms
        else:
            raise ValueError("Invalid reward type. Please choose 'linear', 'quadratic', or 'exponential'.")
        total_cost -= (compressor_cost + violation_penalty)* self.delta_t
        total_cost = total_cost/self.scale
        
        return compressor_cost, violation_penalty, total_cost