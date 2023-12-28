# running environment of DP
import numpy as np
from common.utils import get_driving_cycle, get_acc_limit


class DP_Env:
    def __init__(self, scenario_name):
        # speed list
        self.speed_list = get_driving_cycle(cycle_name=scenario_name)
        self.acc_list = get_acc_limit(self.speed_list, output_max_min=False)
        self.abs_spd_MAX = max(abs(self.speed_list))
        self.abs_acc_MAX = max(abs(max(self.acc_list)), abs(min(self.acc_list)))
        self.time_steps = len(self.speed_list)
        self.trip_length = sum(self.speed_list)/1000     # km
        # state space
        self.state_dim = 1      # SoC
        self.state_increment = 0.005        # this is OK
        self.state_init = 0.5
        self.state_max = 0.55
        self.state_min = 0.01
        self.states = np.arange(self.state_min, self.state_max, self.state_increment)
        self.state_num = len(self.states)
        # action space
        self.action_dim = 1  # P_fc， kW
        self.action_number = 5
        self.actions = np.linspace(0, 1, self.action_number+1, dtype=np.float32)
        # value function
        self.values = np.zeros(self.state_num+1, dtype=np.float32)
        # policy
        self.policy = np.zeros(self.time_steps, dtype=np.float32)


if __name__ == '__main__':
    print("---debug---")
    scenario_name = 'CLTCP_WVUINTER'
    dp_env = DP_Env(scenario_name)
    print('scenario_name: %s'%scenario_name)
    print(dp_env.actions)
    print(60*dp_env.actions)
    print(dp_env.states)
    print('\naction-shape', dp_env.actions.shape)
    print('\nstate-shape', dp_env.states.shape)
    print('\nstate*shape', dp_env.states.shape[0]*dp_env.actions.shape[0])
    print('\ntime-step', dp_env.time_steps)
    print('\nstep %d * state %d * action %d: %d'%
          (dp_env.time_steps, dp_env.states.shape[0], dp_env.actions.shape[0],
           dp_env.states.shape[0]*dp_env.actions.shape[0]*dp_env.time_steps))
    
    
