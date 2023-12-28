import os
import torch
import torch.nn.functional as F
import torch.optim as opt
from network import QNetwork, GaussianPolicy, BetaPolicy

class SAC:
    def __init__(self, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.automatic_entropy_tuning = args.auto_tune
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        
        # soft Q-function
        self.critic = QNetwork(args).to(self.device)
        self.critic_target = QNetwork(args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = opt.Adam(self.critic.parameters())
        
        # policy network
        if args.policy == 'Guassian':
            print("\n---Guassian policy is employed.---\n")
            self.actor = GaussianPolicy(args).to(self.device)
        else:       # 'Beta'
            print("\n---Beta policy is employed.---\n")
            self.actor = BetaPolicy(args).to(self.device)
        self.actor_optimizer = opt.Adam(self.actor.parameters())
        
        # automatic_entropy_tuning
        if self.automatic_entropy_tuning is True:
            self.target_entropy = float(-args.action_dim)
            # self.target_entropy = log(args.action_dim)  # its value is 0, <float>    worse performance
            self.log_alpha = torch.ones(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = opt.Adam([self.log_alpha], lr=args.lr_alpha)
        # device info
        print('actor device: ', list(self.actor.parameters())[0].device)
        print('critic device: ', list(self.critic.parameters())[0].device)
        # print('alpha device: ', list(self.critic.parameters())[0].device)
        
        # learning rate schedulers
        step_size = int(args.max_episodes/10)
        self.scheduler_lr_cr = opt.lr_scheduler.CyclicLR(self.critic_optimizer, base_lr=args.base_lrs,
                                                         max_lr=args.lr_critic, step_size_up=step_size,
                                                         mode="triangular2", cycle_momentum=False)
        self.scheduler_lr_ac = opt.lr_scheduler.CyclicLR(self.actor_optimizer, base_lr=args.base_lrs,
                                                         max_lr=args.lr_actor, step_size_up=step_size,
                                                         mode="triangular2", cycle_momentum=False)
        self.scheduler_lr_al = opt.lr_scheduler.CyclicLR(self.alpha_optimizer, base_lr=args.base_lrs,
                                                         max_lr=args.lr_alpha, step_size_up=step_size,
                                                         mode="triangular2", cycle_momentum=False)
        
        if args.load_or_not is False:
            # create the directory to store the model
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            self.model_path = args.save_dir+'/'+args.scenario_name+'/'+'net_params'
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
        else:
            # load model to evaluate
            load_path = args.load_dir+'/'+args.load_scenario_name+'/'+'net_params'
            actor_pkl = '/actor_params_ep%d.pkl'%args.load_episode
            critic_pkl = '/critic_params_ep%d.pkl'%args.load_episode
            load_a = load_path+actor_pkl
            load_c = load_path+critic_pkl
            if os.path.exists(load_a):
                self.actor.load_state_dict(torch.load(load_a))
                self.critic.load_state_dict(torch.load(load_c))
                print('Agent successfully loaded actor_network: {}'.format(load_a))
                print('Agent successfully loaded critic_network: {}'.format(load_c))
            else:
                print('----Failed to load----')
            if self.automatic_entropy_tuning is True:
                load_alpha = load_path+'/alpha_params_ep%d.pkl'%args.load_episode
                self.alpha_optimizer.load_state_dict(torch.load(load_alpha))
                print('Agent successfully loaded alpha_network: {}'.format(load_alpha))
    
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_((1-self.tau)*target_param.data+self.tau*param.data)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Tensor(1,1)
        if evaluate is False:
            action, _, _ = self.actor.get_action(state)
        else:
            _, _, action = self.actor.get_action(state)
        return action.detach().cpu().numpy()[0]
    
    def learn(self, transition):
        state_batch = transition[0]
        action_batch = transition[1]
        reward_batch = transition[2]
        next_state_batch = transition[3]
        # mask_batch = transition[4]
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # Update the Q-function parameters
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.get_action(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)-self.alpha*next_log_pi
            # next_q_value = reward_batch + (1-mask_batch)*self.gamma*min_qf_next_target
            next_q_value = reward_batch + self.gamma*min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss+qf2_loss
        
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()
        # soft update target Q-function parameters
        self._soft_update_target_network()
        
        # Update policy weights
        action, log_pi, _ = self.actor.get_action(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha*log_pi)-min_qf_pi).mean()
        # minimizing this: Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Adjust temperature
        if self.automatic_entropy_tuning is True:
            alpha_loss = -(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs
        return qf1_loss.item(), qf2_loss.item(), actor_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def save_net(self, episode):
        torch.save(self.actor.state_dict(), self.model_path+'/actor_params_ep%d.pkl'%episode)
        torch.save(self.critic.state_dict(), self.model_path+'/critic_params_ep%d.pkl'%episode)
        if self.automatic_entropy_tuning is True:
            torch.save(self.alpha_optimizer.state_dict(), self.model_path+'/alpha_params_ep%d.pkl'%episode)
         
    def lr_scheduler(self):
        lrcr = self.scheduler_lr_cr.get_last_lr()[0]
        lrac = self.scheduler_lr_ac.get_last_lr()[0]
        lral = self.scheduler_lr_al.get_last_lr()[0]
        self.scheduler_lr_cr.step()
        self.scheduler_lr_ac.step()
        self.scheduler_lr_al.step()
        return lrcr, lrac, lral