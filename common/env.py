from common.utils import get_driving_cycle, get_acc_limit
from common.agentEMS import EMS

class Env:
    """ environment for EMS"""
    
    def __init__(self, args):
        self.args = args
        self.speed_list = get_driving_cycle(cycle_name=args.scenario_name)
        self.acc_list = get_acc_limit(self.speed_list, output_max_min=False)
        self.abs_spd_MAX = max(abs(self.speed_list))
        self.abs_acc_MAX = max(abs(max(self.acc_list)), abs(min(self.acc_list)))
        
        self.agent = EMS(args.w_soc, args.soc0, args.MODE, self.abs_spd_MAX, self.abs_acc_MAX)
        self.obs_num = self.agent.obs_num
        self.action_num = self.agent.action_num
    
    def reset(self):
        return self.agent.reset_obs()
    
    def step(self, action, episode_step):
        car_spd = self.speed_list[episode_step]
        car_acc = self.acc_list[episode_step]
        # epi_next = int(min(episode_step+1, self.args.episode_steps-1))
        # car_spd_next = self.speed_list[epi_next]
        # car_acc_next = self.acc_list[epi_next]
        
        obs = self.agent.execute(action, car_spd, car_acc)
        reward = self.agent.get_reward()
        done = self.agent.get_done()
        info = self.agent.get_info()
        return obs, reward, done, info

def make_env(args):
    env = Env(args)
    args.obs_dim = env.agent.obs_num
    args.action_dim = env.agent.action_num
    args.episode_steps = len(env.speed_list)  # cycle length, be equal to args.episode_steps
    args.abs_spd_MAX = env.abs_spd_MAX
    args.abs_acc_MAX = env.abs_acc_MAX
    return env, args
