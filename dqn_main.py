from tqdm import tqdm
import numpy as np
import torch
import os
import scipy.io as scio
from common.dqn_model import DQN_model, Memory
from common.arguments import get_args
from common.agentEMS import EMS
from common.utils import get_driving_cycle, get_acc_limit

def main_EMS(args, speed_list, acc_list, episodes=400):
    save_path = args.save_dir+"_DQN_"+args.MODE+'/' \
                +args.scenario_name+"_w%d"%args.w_soc+'_' \
                +"LR%.0e"%args.lr_DQN+ '_' +args.file_v
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path_episode = save_path+'/episode_data'
    if not os.path.exists(save_path_episode):
        os.makedirs(save_path_episode)
    
    abs_spd_MAX = max(abs(speed_list))
    abs_acc_MAX = max(abs(max(acc_list)), abs(min(acc_list)))
    action_num = 60
    action_space = np.linspace(0, 1, action_num, dtype=np.float32)
    ems = EMS(args.w_soc, args.soc0, args.MODE, abs_spd_MAX, abs_acc_MAX)
    memory = Memory(memory_size=args.buffer_size, batch_size=args.batch_size)
    dqn_agent = DQN_model(args, s_dim=ems.obs_num, a_dim=action_num)
    
    average_reward = []  # average_reward of each episode
    average_loss = []
    DONE = {}
    lr = []
    epsilon_list = []
    h2_100_list = []
    eq_h2_100_list = []  # equivalent hydrogen consumption per 100 km
    money_100_list = []  # money spent per 100 km
    FCS_SoH = []
    Batt_SoH = []
    SOC = []
    
    initial_epsilon = 1.0
    finial_epsilon = 0.2
    epsilon_decent = (initial_epsilon-finial_epsilon)/150
    epsilon = initial_epsilon
    
    SPD_LIST = speed_list
    ACC_LIST = acc_list
    episode_step_num = SPD_LIST.shape[0]
    MILE = np.sum(SPD_LIST)/1000
    print('mileage: %.3fkm'%MILE)
    
    for episode in tqdm(range(episodes)):
        state = ems.reset_obs()  # ndarray
        rewards = []
        loss = []
        info = []
        episode_info = {'T_mot': [], 'W_mot': [], 'mot_eff': [], 'P_mot': [],
                        'P_fc': [], 'P_fce': [], 'fce_eff': [], 'FCS_SOH': [],
                        'P_dcdc': [], 'dcdc_eff': [], 'FCS_De': [], 'travel': [],
                        'd_s_s': [], 'd_low': [], 'd_high': [], 'd_l_c': [],
                        'EMS_reward': [], 'soc_cost': [], 'h2_equal': [], 'h2_fcs': [],
                        'money_cost': [], 'h2_money': [], 'batt_money': [], 'fcs_money': [],
                        'SOC': [], 'SOH': [], 'I': [], 'I_c': [], 'money_cost_real': [],
                        'cell_OCV': [], 'cell_Vt': [], 'cell_V_3': [],
                        'cell_power_out': [], 'P_batt': [], 'tep_a': [], 'dsoh': []}
        for episode_step in range(episode_step_num):
            with torch.no_grad():
                action_id, epsilon_using = dqn_agent.e_greedy_action(state, epsilon)
            action = [action_space[action_id]]  # [float]
            spd = SPD_LIST[episode_step]
            acc = ACC_LIST[episode_step]
            next_state = ems.execute(action, spd, acc)
            reward = ems.get_reward()
            done = ems.get_done()
            info = ems.get_info()
            
            memory.store_trasition(state, action_id, reward, next_state)
            state = next_state
            
            rewards.append(reward)
            if done and episode not in DONE.keys():
                print('\nSOC failure in step %d of episode %d'%(episode_step, episode))
                DONE.update({episode: episode_step})
                # break
            for key in episode_info.keys():
                episode_info[key].append(info[key])
            
            if memory.current_size > 100*args.batch_size:
                minibatch = memory.uniform_sample()
                dqn_agent.train(minibatch)
                loss.append(dqn_agent.loss)
            
            # end of an episode: sava model params,
            if episodes-episode <= 20 and episode_step+1 == episode_step_num:
                dqn_agent.save_model(save_path, episode)
                datadir = save_path_episode+'/data_ep%d.mat'%episode
                scio.savemat(datadir, mdict=episode_info)
        # end of one episode: save data, print info
        # show episode data
        travel = info['travel']/1000  # km
        h2 = sum(episode_info['h2_fcs'])  # g
        eq_h2 = sum(episode_info['h2_equal'])  # g
        money = sum(episode_info['money_cost_real'])  # RMB
        h2_100 = h2/travel*100
        equal_h2_100 = eq_h2/travel*100
        m_100 = money/travel*100
        h2_100_list.append(h2_100)
        eq_h2_100_list.append(equal_h2_100)
        money_100_list.append(m_100)
        # print
        soc = info['SOC']
        fcs_soh = info['FCS_SOH']
        bat_soh = info['SOH']
        FCS_SoH.append(fcs_soh)
        Batt_SoH.append(bat_soh)
        SOC.append(soc)
        print('\nepi %d: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f'
              %(episode, travel, soc, fcs_soh, bat_soh))
        print('epi %d: H2_100km %.1fg, eq_H2_100km %.1fg, money_100km ￥%.2f'%
              (episode, h2_100, equal_h2_100, m_100))
        # save loss and reward on average
        ep_r = np.mean(rewards)
        ep_loss = np.mean(loss)
        average_reward.append(ep_r)
        average_loss.append(ep_loss)
        lr0 = dqn_agent.scheduler_lr.get_last_lr()[0]
        lr.append(lr0)
        epsilon_list.append(epsilon_using)
        print('epi %d: reward %.6f, loss %.6f, epsilon %.6f, lr %.6f'
              %(episode, ep_r, ep_loss, epsilon_using, lr0))
        epsilon -= float(epsilon_decent)
        epsilon = max(epsilon, finial_epsilon)
        dqn_agent.scheduler_lr.step()
    
    scio.savemat(save_path+'/lr.mat', mdict={'lr': lr})
    scio.savemat(save_path+'/epsilon_list.mat', mdict={'epsilon': epsilon_list})
    scio.savemat(save_path+'/loss.mat', mdict={'loss': average_loss})
    scio.savemat(save_path+'/reward.mat', mdict={'reward': average_reward})
    scio.savemat(save_path+'/h2.mat', mdict={'h2': h2_100_list})
    scio.savemat(save_path+'/eq_h2.mat', mdict={'eq_h2': eq_h2_100_list})
    scio.savemat(save_path+'/money.mat', mdict={'money': money_100_list})
    scio.savemat(save_path+'/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SoH})
    scio.savemat(save_path+'/Batt_SOH.mat', mdict={'Batt_SOH': Batt_SoH})
    scio.savemat(save_path+'/SOC.mat', mdict={'SOC': SOC})
    
    print('buffer counter:', memory.counter)
    print('buffer current size:', memory.current_size)
    print('replay ratio: %.3f'%(memory.counter/memory.current_size)+'\n')
    print('done:', DONE)

if __name__ == '__main__':
    args = get_args()
    speed_list = get_driving_cycle(cycle_name=args.scenario_name)
    acc_list = get_acc_limit(speed_list, output_max_min=False)
    
    seed = np.random.randint(100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Random seeds have been set to %d!"%seed)
    print('cycle name: ', args.scenario_name)
    
    main_EMS(args, speed_list, acc_list)