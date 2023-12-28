"""
Dynamic Programming, energy management
"""
import os
import time
import numpy as np
from tqdm import tqdm
import scipy.io as scio
from common.arguments import get_args
from common.DP_env import DP_Env
from common.agentEMS import EMS
from common.DP_EMS_agent import DP_EMS_agent

class DP_brain:
    def __init__(self, env, AgentEMS, DP_EMS_Agent):
        self.EMS_Agent = AgentEMS
        self.DP_EMS_agent = DP_EMS_Agent
        self.env = env
        self.gamma = 0.9
        self.states = self.env.states
        self.actions = self.env.actions
        self.policy_as_action_id = np.zeros(self.env.time_steps, dtype=np.float32)
        self.policy_as_reward_id = np.zeros(self.env.time_steps, dtype=np.float32)
        self.optimal_action_id_table = np.zeros((len(self.states), self.env.time_steps), dtype=np.int32)
        self.optimal_reward_table = np.zeros((len(self.states), self.env.time_steps), dtype=np.float32)
        self.optimal_reward_id_table = np.zeros(self.optimal_reward_table.shape, dtype=np.int32)
        # self.deltas = np.zeros(self.optimal_action_table.shape, dtype=np.float32)
        self.info_dict = {'T_mot': [], 'W_mot': [], 'mot_eff': [], 'P_mot': [],
                          'P_fc': [], 'P_fce': [], 'fce_eff': [], 'FCS_SOH': [],
                          'P_dcdc': [], 'dcdc_eff': [], 'FCS_De': [], 'travel': [],
                          'd_s_s': [], 'd_low': [], 'd_high': [], 'd_l_c': [],
                          'EMS_reward': [], 'soc_cost': [], 'h2_equal': [], 'h2_fcs': [],
                          'money_cost': [], 'h2_money': [], 'batt_money': [], 'fcs_money': [],
                          'SOC': [], 'SOH': [], 'I': [], 'I_c': [], 'money_cost_real': [],
                          'cell_OCV': [], 'cell_Vt': [], 'cell_V_3': [],
                          'cell_power_out': [], 'P_batt': [], 'tep_a': [], 'dsoh': []}
    
    def DP_backward(self):
        for time_step in tqdm(range(self.env.time_steps-2, -1, -1)):
            # [3438, 3437, ... ,0] [time_steps-1, time_steps-2, ... , 0]
            car_spd = self.env.speed_list[time_step]
            car_acc = self.env.acc_list[time_step]
            # print('\nstep: %d, car_spd: %.2f, car_acc: %.2f' % (time_step, car_spd, car_acc))
            for state_id, state in enumerate(self.states):
                state_action_money = np.zeros(len(self.actions), dtype=np.float32)+10000
                # print(state_action_money)
                state_action_reward = np.zeros(len(self.actions), dtype=np.float32)-10000
                for action_id, action in enumerate(self.actions):
                    SOC_new = self.DP_EMS_agent.execute(action=action, car_spd=car_spd, car_acc=car_acc, soc=state)
                    DP_EMS_reward, money_cost = self.DP_EMS_agent.get_reward(SOC_new)
                    state_action_money[action_id] = money_cost   # score every action
                    state_action_reward[action_id] = DP_EMS_reward
                # choose the action-id with the highest score (minimum cost)
                self.optimal_action_id_table[state_id, time_step] = np.argmin(state_action_money)
                self.optimal_reward_table[state_id, time_step] = np.max(state_action_reward)
                self.optimal_reward_id_table[state_id, time_step] = np.argmax(state_action_reward)
            self.optimal_reward_table[:, time_step] +=\
                self.gamma *self.optimal_reward_table[:, time_step+1]
         
    def execute(self, car_spd, car_acc, P_FC):
        obs = self.EMS_Agent.execute(action=[P_FC], car_spd=car_spd, car_acc=car_acc)
        reward = self.EMS_Agent.get_reward()
        info = self.EMS_Agent.get_info()
        soc_new = obs[0]
        return soc_new, reward, info
    
    def find_s_idx(self, SOC_new):
        delta_list = abs(SOC_new-self.env.states)
        near_s_id = np.argmin(delta_list)
        near_s = self.states[near_s_id]
        return near_s_id, near_s
 
    def get_forward_policy(self):
        near_s0_list = []
        self.EMS_Agent.reset_obs()
        s0 = self.env.state_init
        s0_idx, near_s0 = self.find_s_idx(s0)
        info = {}
        for step in range(self.env.time_steps):
            near_s0_list.append(near_s0)
            car_spd = self.env.speed_list[step]
            car_acc = self.env.acc_list[step]
            # get action
            act_id = self.optimal_action_id_table[s0_idx, step]
            act1 = self.actions[act_id]
            self.policy_as_action_id[step] = act1
            act_id = self.optimal_reward_id_table[s0_idx, step]
            act2 = self.actions[act_id]
            self.policy_as_reward_id[step] = act2
            act = act2
            
            s_new, reward, info = self.execute(car_spd, car_acc, act)
            for key in self.info_dict.keys():
                self.info_dict[key].append(info[key])
            
            s_idx, near_s = self.find_s_idx(s_new)  # id of new state
            near_s0 = near_s
            s0_idx = s_idx
            
        self.info_dict.update({'policy_as_action': self.policy_as_action_id.tolist(),
                               'policy_as_reward': self.policy_as_reward_id.tolist(),
                               'near_s0_list': near_s0_list})
        print("---dynamic programming finished!---")
        # show data
        travel = info['travel']/1000  # km
        h2 = sum(self.info_dict['h2_fcs'])  # g
        eq_h2 = sum(self.info_dict['h2_equal'])  # g
        money = sum(self.info_dict['money_cost'])  # RMB
        h2_100 = h2/travel*100
        equal_h2_100 = eq_h2/travel*100
        m_100 = money/travel*100
        self.info_dict.update({'money_100': m_100, 'eq_h2_100': equal_h2_100})
        # print
        soc = info['SOC']
        fcs_soh = info['FCS_SOH']
        bat_soh = info['SOH']
        print('\nDP-EMS: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f'
              %(travel, soc, fcs_soh, bat_soh))
        print('DP-EMS: H2_100km %.1fg, eq_H2_100km %.1fg, money_100km ￥%.2f'
              %(h2_100, equal_h2_100, m_100))
 
 
if __name__ == "__main__":
    strat_tiem = time.time()
    args = get_args()
    scenario = args.scenario_name  # CTUDC, WVU, JN
    dp_env = DP_Env(scenario)
    print('scenario name: %s'%scenario)
    print('\nstep %d * state %d * action %d: %d'%
          (dp_env.time_steps, dp_env.states.shape[0], dp_env.actions.shape[0],
           dp_env.states.shape[0]*dp_env.actions.shape[0]*dp_env.time_steps))
    EMSAgent = EMS(w_soc=args.w_soc, soc0=0.5, SOC_MODE=args.MODE,
                   abs_spd_MAX=dp_env.abs_spd_MAX, abs_acc_MAX=dp_env.abs_acc_MAX)
    DP_EMS_Agent = DP_EMS_agent(w_soc=args.w_soc, gamma=0.9)
    DP_brain = DP_brain(dp_env, EMSAgent, DP_EMS_Agent)
    DP_brain.DP_backward()
    # save data dir
    datadir = './DP_result/' + scenario + '_w%d' % args.w_soc + '_'+args.file_v
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    # save backward data
    dataname_back1 = '/action_id.mat'
    scio.savemat(datadir+dataname_back1, mdict={'action_id': DP_brain.optimal_action_id_table})
    dataname_back2 = '/reward_id.mat'
    scio.savemat(datadir+dataname_back2, mdict={'reward_id': DP_brain.optimal_reward_id_table})
    dataname_back3 = '/optimal_reward.mat'
    scio.savemat(datadir+dataname_back3, mdict={'optimal_reward': DP_brain.optimal_reward_table})
    end_claculate = time.time()
    calculation_time = end_claculate-strat_tiem
    print("\ntime for calculation: %.2fs"%calculation_time)
    # forward
    DP_brain.get_forward_policy()
    dataname = '/DP_EMS_info.mat'
    scio.savemat(datadir+dataname, mdict={'DP_EMS_info': DP_brain.info_dict})
    
    end_time = time.time()
    spent_time = end_time-end_claculate
    print("\ntime for forward_policy: %.2fs"%spent_time)
    