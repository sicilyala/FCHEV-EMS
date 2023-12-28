import sys
import warnings
from runner import Runner
from arguments import get_args
from env import make_env
from evaluate import Evaluator
from utils import Logger

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = get_args()
    args.save_dir = args.save_dir + "_" + args.DRL + "_" + args.MODE + "_" + args.policy
    args.log_dir = args.log_dir + "_" + args.DRL + "_" + args.MODE + "_" + args.policy
    args.eva_dir = args.eva_dir + "_" + args.DRL + "_" + args.MODE + "_" + args.policy
    env, args = make_env(args)
    if args.evaluate:
        args.scenario_name = args.load_dir+'/'+args.load_scenario_name+'_%d'%args.load_episode
        sys.stdout = Logger(filepath=args.eva_dir+"/"+args.scenario_name+"/", filename='evaluate_log.log')
        print('max_episodes: ', args.evaluate_episode)
    else:
        args.scenario_name = args.scenario_name + "_w%d"%args.w_soc + "_LR%.0e"%args.lr_critic + '_' + args.file_v
        sys.stdout = Logger(filepath=args.save_dir+"/"+args.scenario_name+"/", filename='train_log.log')
        print('\nweight coefficient: w_soc = %.1f' % args.w_soc)
        print('max_episodes: ', args.max_episodes)
    print('cycle name: ', args.scenario_name)
    print('episode_steps: ', args.episode_steps)
    print('abs_spd_MAX: %.3f m/s' % args.abs_spd_MAX)
    print('abs_acc_MAX: %.3f m/s2' % args.abs_acc_MAX)
    print("DRL method: ", args.DRL)
    print('obs_dim: ', args.obs_dim)
    print('action_dim: ', args.action_dim)
    print('critic initial learning rate: %.0e' % args.lr_critic)
    print('actor initial learning rate: %.0e'%args.lr_actor)
    if args.DRL == 'SAC':
        print('alpha initial learning rate: %.0e'%args.lr_alpha)
    print('initial SOC: ', args.soc0)
    print('SOC-MODE: ', args.MODE)
    
    if args.evaluate:
        print("\n-----Start evaluating!-----")
        evaluator = Evaluator(args, env)
        evaluator.evaluate()
        print("-----Evaluating is finished!-----")
        print('-----Data saved in: <%s>-----'%(args.eva_dir+"/"+args.scenario_name))
    else:
        print("\n-----Start training-----")
        runner = Runner(args, env)
        runner.set_seed()
        if args.DRL == 'SAC':
            runner.run_SAC()
        elif args.DRL == 'DDPG':
            runner.run_DDPG()
        else:
            print("\n[ERROR]: No such DRL method in this program! Exit Now!\n")
        runner.memory_info()
        print("-----Training is finished!-----")
        print('-----Data saved in: <%s>-----'%(args.save_dir+"/"+args.scenario_name))
