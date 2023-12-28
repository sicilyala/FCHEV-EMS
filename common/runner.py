import datetime
import os
import numpy as np
from numpy.random import normal  # normal distribution
import scipy.io as scio
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from memory import MemoryBuffer
from sac import SAC
from ddpg import DDPG

class Runner:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.buffer = MemoryBuffer(args)
        self.SAC_agent = SAC(args)
        self.DDPG_agent = DDPG(args)
        # configuration
        self.episode_num = args.max_episodes
        self.episode_step = args.episode_steps
        self.DONE = {}
        self.save_path = self.args.save_dir+'/'+self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path_episode = self.save_path+'/episode_data'
        if not os.path.exists(self.save_path_episode):
            os.makedirs(self.save_path_episode)
        # random seed
        if self.args.random_seed:
            self.seed = np.random.randint(100)
        else:
            self.seed = 93
        # tensorboard
        fileinfo = args.scenario_name
        self.writer = SummaryWriter(args.log_dir+'/{}_{}_{}_seed{}'.format
                                    (datetime.datetime.now().strftime("%m-%d_%H-%M"),
                                     fileinfo, self.args.DRL, self.seed))
        
    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        print("Random seeds have been set to %d !\n"%self.seed)
    
    def run_SAC(self):
        average_reward = []  # average_reward of each episode
        c_loss_1 = []
        c_loss_2 = []
        a_loss = []
        en_loss = []
        h2_100_list = []
        eq_h2_100_list = []  # equivalent hydrogen consumption per 100 km
        money_100_list = []  # money spent per 100 km
        FCS_SoH = []
        Batt_SoH = []
        SOC = []
        lr_recorder = {'lrcr': [], 'lrac': [], 'lral': []}
        updates = 0  # for tensorboard counter
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            episode_reward = []
            c_loss_1_one_ep = []
            c_loss_2_one_ep = []
            a_loss_one_ep = []
            en_loss_one_ep = []
            alpha_value_ep = []
            info = []
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
            
            for episode_step in range(self.episode_step):
                with torch.no_grad():
                    action = self.SAC_agent.select_action(state)
                state_next, reward, done, info = self.env.step(action, episode_step)
                # when done is True: unsafe, stop
                self.buffer.store(state, action, reward, state_next, done)
                if done and episode not in self.DONE.keys():
                    print('\nfailure in step %d of episode %d'%(episode_step, episode))
                    self.DONE.update({episode: episode_step})
                    # break
                state = state_next
                # save data
                for key in episode_info.keys():
                    episode_info[key].append(info[key])
                episode_reward.append(reward)
                # learn
                if self.buffer.currentSize >= 10*self.args.batch_size:
                    transition = self.buffer.random_sample()
                    critic_loss_1, critic_loss_2, actor_loss, alpha_loss, alpha = self.SAC_agent.learn(transition)
                    # save to tensorboard
                    self.writer.add_scalar('loss/critic_1', critic_loss_1, updates)
                    self.writer.add_scalar('loss/critic_2', critic_loss_2, updates)
                    self.writer.add_scalar('loss/actor', actor_loss, updates)
                    self.writer.add_scalar('loss/alpha_loss', alpha_loss, updates)
                    self.writer.add_scalar('entropy/alpha_value', alpha, updates)
                    self.writer.add_scalar('reward/step_reward', reward, updates)
                    updates += 1
                    # save in .mat
                    c_loss_1_one_ep.append(critic_loss_1)
                    c_loss_2_one_ep.append(critic_loss_2)
                    a_loss_one_ep.append(actor_loss)
                    en_loss_one_ep.append(alpha_loss)
                    alpha_value_ep.append(alpha)
                # save data in .mat
                # if episode_step+1 == self.episode_step:
                if self.episode_num-episode<=400 and episode_step+1 == self.episode_step:
                    # only save the last 10 episode
                    # save alpha value of each time step in each episode
                    episode_info.update({'alpha': alpha_value_ep})
                    # save network parameters
                    self.SAC_agent.save_net(episode)
                    # save all data in one episode info
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    scio.savemat(datadir, mdict=episode_info)
            # lr sheduler
            lrcr, lrac, lral = self.SAC_agent.lr_scheduler()
            lr_recorder['lrcr'].append(lrcr)
            lr_recorder['lrac'].append(lrac)
            lr_recorder['lral'].append(lral)
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
            print('\nepi %d: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f'
                  % (episode, travel, soc, fcs_soh, bat_soh))
            print('epi %d: H2_100km %.1fg, eq_H2_100km %.1fg, money_100km ￥%.2f'%
                  (episode, h2_100, equal_h2_100, m_100))
            # save loss and reward on the average
            ep_r = np.mean(episode_reward)
            ep_c1 = np.mean(c_loss_1_one_ep)
            ep_c2 = np.mean(c_loss_2_one_ep)
            ep_a = np.mean(a_loss_one_ep)
            ep_en = np.mean(en_loss_one_ep)
            print('epi %d: ep_r %.3f, c-loss1 %.4f, c-loss2 %.4f, a-loss %.4f, en-loss %.4f'
                  % (episode, ep_r, ep_c1, ep_c2, ep_a, ep_en))
            print('epi %d: lr_critic %.6f, lr_actor %.6f, lr_alpha %.6f' % (episode, lrcr, lrac, lral))
            average_reward.append(ep_r)
            c_loss_1.append(ep_c1)
            c_loss_2.append(ep_c2)
            a_loss.append(ep_a)
            en_loss.append(ep_en)
            FCS_SoH.append(fcs_soh)
            Batt_SoH.append(bat_soh)
            SOC.append(soc)
        
        scio.savemat(self.save_path+'/reward.mat', mdict={'reward': average_reward})
        scio.savemat(self.save_path+'/critic_loss.mat', mdict={'c_loss_1': c_loss_1, 'c_loss_2': c_loss_2})
        scio.savemat(self.save_path+'/actor_loss.mat', mdict={'a_loss': a_loss})
        scio.savemat(self.save_path+'/entropy_loss.mat', mdict={'en_loss': en_loss})
        scio.savemat(self.save_path+'/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path+'/h2.mat', mdict={'h2': h2_100_list})
        scio.savemat(self.save_path+'/eq_h2.mat', mdict={'eq_h2': eq_h2_100_list})
        scio.savemat(self.save_path+'/money.mat', mdict={'money': money_100_list})
        scio.savemat(self.save_path+'/FCS_SOH.mat', mdict={'FCS_SOH': FCS_SoH})
        scio.savemat(self.save_path+'/Batt_SOH.mat', mdict={'Batt_SOH': Batt_SoH})
        scio.savemat(self.save_path+'/SOC.mat', mdict={'SOC': SOC})

    def run_DDPG(self):
        noise_decrease = False
        noise_rate = self.args.noise_rate
        average_reward = []  # average_reward of each episode
        c_loss = []
        a_loss = []
        h2_100_list = []
        eq_h2_100_list = []  # equivalent hydrogen consumption per 100 km
        money_100_list = []  # money spent per 100 km
        FCS_SoH = []
        Batt_SoH = []
        SOC = []
        lr_recorder = {'lrcr': [], 'lrac': []}
        noise = []
        updates = 0  # for tensorboard counter
        for episode in tqdm(range(self.episode_num)):
            state = self.env.reset()  # reset the environment
            if noise_decrease:
                noise_rate *= self.args.noise_discount_rate
            episode_reward = []
            c_loss_one_ep = []
            a_loss_one_ep = []
            info = []
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
        
            for episode_step in range(self.episode_step):
                with torch.no_grad():
                    raw_action = self.DDPG_agent.select_action(state)
                action = np.clip(normal(raw_action, noise_rate), -1, 1)
                state_next, reward, done, info = self.env.step(action, episode_step)
                self.buffer.store(state, action, reward, state_next, done)
                if done and episode not in self.DONE.keys():
                    print('failure in step %d of episode %d'%(episode_step, episode))
                    self.DONE.update({episode: episode_step})
                    # break
                state = state_next
                # save data
                for key in episode_info.keys():
                    episode_info[key].append(info[key])
                episode_reward.append(reward)
                # learn
                if self.buffer.currentSize >= 10*self.args.batch_size:
                    noise_decrease = True
                    transition = self.buffer.random_sample()
                    critic_loss, actor_loss = self.DDPG_agent.train(transition)
                    # save to tensorboard
                    self.writer.add_scalar('loss/critic', critic_loss, updates)
                    self.writer.add_scalar('loss/actor', actor_loss, updates)
                    self.writer.add_scalar('reward/step_reward', reward, updates)
                    updates += 1
                    # save in .mat
                    c_loss_one_ep.append(critic_loss)
                    a_loss_one_ep.append(actor_loss)
                # save data in .mat     # only save the last 10 episode
                if self.episode_num-episode<=50 and episode_step+1 == self.episode_step:
                    self.DDPG_agent.save_net(episode)
                    datadir = self.save_path_episode+'/data_ep%d.mat'%episode
                    scio.savemat(datadir, mdict=episode_info)
            # lr sheduler
            lra, lrc = self.DDPG_agent.lr_scheduler()
            lr_recorder['lrcr'].append(lrc)
            lr_recorder['lrac'].append(lra)
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
            print('\nepi %d: travel %.3fkm, SOC %.4f, FCS-SOH %.6f, Bat-SOH %.6f'
                  % (episode, travel, soc, fcs_soh, bat_soh))
            print('epi %d: H2_100km %.1fg, eq_H2_100km %.1fg, money_100km ￥%.2f'%
                  (episode, h2_100, equal_h2_100, m_100))
            # save loss and reward on the average
            ep_r = np.mean(episode_reward)
            ep_c = np.mean(c_loss_one_ep)
            ep_a = np.mean(a_loss_one_ep)
            print('epi %d: ep_r %.3f, c-loss %.4f, a-loss %.4f'
                  % (episode, ep_r, ep_c, ep_a))
            print('epi %d: noise_rate %.6f, lr_critic %.6f, lr_actor %.6f'
                  % (episode, noise_rate, lra, lrc))
            average_reward.append(ep_r)
            c_loss.append(ep_c)
            a_loss.append(ep_a)
            FCS_SoH.append(fcs_soh)
            Batt_SoH.append(bat_soh)
            SOC.append(soc)
            noise.append(noise_rate)
    
        scio.savemat(self.save_path+'/reward.mat', mdict={'reward': average_reward})
        scio.savemat(self.save_path+'/critic_loss.mat', mdict={'c_loss': c_loss})
        scio.savemat(self.save_path+'/actor_loss.mat', mdict={'a_loss': a_loss})
        scio.savemat(self.save_path+'/lr_recorder.mat', mdict=lr_recorder)
        scio.savemat(self.save_path+'/h2.mat', mdict={'h2': h2_100_list})
        scio.savemat(self.save_path+'/eq_h2.mat', mdict={'eq_h2': eq_h2_100_list})
        scio.savemat(self.save_path+'/money.mat', mdict={'money': money_100_list})
        scio.savemat(self.save_path+'/FCS_SoH.mat', mdict={'FCS_SoH': FCS_SoH})
        scio.savemat(self.save_path+'/Batt_SoH.mat', mdict={'Batt_SoH': Batt_SoH})
        scio.savemat(self.save_path+'/SOC.mat', mdict={'SOC': SOC})
        scio.savemat(self.save_path+'/noise.mat', mdict={'noise': noise})
        
    def memory_info(self):
        print('\nbuffer counter:', self.buffer.counter)
        print('buffer current size:', self.buffer.currentSize)
        print('replay ratio: %.3f'%(self.buffer.counter/self.buffer.currentSize))
        print('failure:', self.DONE)