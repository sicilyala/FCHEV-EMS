from tqdm import tqdm
import os
import torch
import numpy as np
import time
import scipy.io as scio
from common.sac import SAC

class Evaluator:
    def __init__(self, args, env):
        self.args = args
        self.eva_episode = args.evaluate_episode
        self.episode_step = args.episode_steps
        self.env = env
        self.DRL_agent = SAC(args)      # TODO 定义不同的DRL智能体
    
        self.save_path = self.args.eva_dir+'/'+self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
 
    def evaluate(self):
        average_reward = []  # average_reward of each episode
        fuel_100 = []  # equivalent hydrogen consumption per 100 km
        real_h2_100 = []
        money_100 = []  # money spent per 100 km
        Batt_SoH = []
        FCS_SOH = []
        for episode in tqdm(range(self.eva_episode)):
            state = self.env.reset()  # reset the environment
            setp_reward = []
            # data being saved in .mat
            episode_info = {'T_mot': [], 'W_mot': [], 'mot_eff': [], 'P_mot': [],
                            'P_fc': [], 'P_fce': [], 'fce_eff': [], 'FCS_SOH': [],
                            'P_dcdc': [], 'dcdc_eff': [], 'FCS_De': [], 'travel': [],
                            'd_s_s': [], 'd_low': [], 'd_high': [], 'd_l_c': [],
                            'EMS_reward': [], 'soc_cost': [], 'h2_equal': [], 'h2_fcs': [],
                            'money_cost': [], 'h2_money': [], 'batt_money': [], 'fcs_money': [],
                            'SOC': [], 'SOH': [], 'I': [], 'I_c': [], 'money_cost_real': [],
                            'cell_OCV': [], 'cell_Vt': [], 'cell_V_3': [],
                            'cell_power_out': [], 'P_batt': [], 'tep_a': [], 'dsoh': []}
            start_time = time.time()
            for episode_step in range(self.episode_step):
                with torch.no_grad():
                    raw_action = self.DRL_agent.select_action(state, evaluate=True)
                action = raw_action
                state_next, reward, done, info = self.env.step(action, episode_step)
                state = state_next
                # save data
                for key in episode_info.keys():
                    episode_info[key].append(info[key])
                setp_reward.append(reward)
                # save data in .mat
                if episode_step+1 == self.episode_step:
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    scio.savemat(datadir, mdict=episode_info)
                    # print
                    f_travel = info['travel']/1000
                    h2 = sum(episode_info['h2_fcs'])  # g
                    eq_h2 = sum(episode_info['h2_equal'])  # g
                    money = sum(episode_info['money_cost'])  # RMB
                    h2_100 = h2/f_travel*100
                    eq_h2_100 = eq_h2/f_travel*100
                    m_100 = money/f_travel*100
                    fuel_100.append(eq_h2_100)
                    real_h2_100.append(h2_100)
                    money_100.append(m_100)
                    soc = info['SOC']
                    bat_soh = info['SOH']
                    fcs_soh = info['FCS_SOH']
                    Batt_SoH.append(bat_soh)
                    FCS_SOH.append(fcs_soh)
                    print('\nepi %d: travel %.3fkm, SOC %.4f, Bat-SOH %.6f, FCS-SOH %.6f'
                          %(episode, f_travel, soc, bat_soh, fcs_soh))
                    print('epi %d: h2_100km %.2fg, money_100km ￥%.2f'%(episode, eq_h2_100, m_100))
                
            end_time = time.time()
            spent_time = end_time-start_time
            # save reward
            ep_r_mean = np.mean(setp_reward)
            average_reward.append(ep_r_mean)
            # print
            print('episode %d: reward %.3f, time spent: %.3fs'
                  %(episode, ep_r_mean, spent_time))
    
        scio.savemat(self.save_path+'/reward.mat', mdict={'reward': average_reward})
        scio.savemat(self.save_path+'/eq_h2_100.mat', mdict={'eq_h2_100': fuel_100})
        scio.savemat(self.save_path+'/money.mat', mdict={'money': money_100})
        scio.savemat(self.save_path+'/Batt_SoH.mat', mdict={'Batt_SoH': Batt_SoH})
        scio.savemat(self.save_path+'/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SOH})
        scio.savemat(self.save_path+'/h2_100.mat', mdict={'real_h2_100': real_h2_100})
    