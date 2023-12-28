import argparse

def get_args():
    parser = argparse.ArgumentParser("Soft Actor Critic Implementation")
    # Environment
    parser.add_argument("--max_episodes", type=int, default=400, help="number of episodes ")
    parser.add_argument("--episode_steps", type=int, default=None, help="number of time steps in a single episode")
    # Core training parameters
    parser.add_argument("--lr_critic", type=float, default=0.001, help="learning rate of critic")
    parser.add_argument("--lr_actor", type=float, default=0.0001, help="learning rate of actor")
    parser.add_argument("--lr_alpha", type=float, default=0.0001, help="learning rate of alpha")
    parser.add_argument("--lr_DQN", type=float, default=0.0005, help="learning rate of alpha")
    parser.add_argument("--base_lrs", type=float, default=5e-5, help="base_lr of CyclicLR")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Temperature parameter α determines the relative importance of the entropy term against the reward')
    parser.add_argument('--auto_tune', type=bool, default=True)
    parser.add_argument("--tau", type=float, default=0.005, help="parameter for updating the target network")
    # for DDPG
    parser.add_argument("--noise_rate", type=float, default=0.125,
                        help="initial noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--noise_discount_rate", type=float, default=0.999)
    # memory buffer
    parser.add_argument("--buffer_size", type=int, default=int(4e4),
                        help="number of transitions can be stored in buffer")
    # 200ep: 3e4 for CLTCP_WVUSUB_WVUINTER,  1e4 for WVUCITY_HWFET
    parser.add_argument("--batch_size", type=int, default=64, help="number of episodes to optimize at the same time")
    # random seeds
    parser.add_argument("--random_seed", type=bool, default=True)
    # device
    parser.add_argument("--cuda", type=bool, default=False, help="True for GPU, False for CPU")
    # method
    parser.add_argument("--DRL", type=str, default='SAC', help="SAC or DDPG")
    # for SAC
    parser.add_argument("--policy", type=str, default='Beta', help="Guassian or Beta")
    # about EMS, weight coefficient
    parser.add_argument("--w_soc", type=float, default=100, help="weight coefficient for SOC reward")
    parser.add_argument("--soc0", type=float, default=0.5, help="initial value of SOC")
    parser.add_argument("--MODE", type=str, default='CS', help="CS or CD, charge-sustain or charge-depletion")
    # save model under training     # Standard_ChinaCity  Standard_IM240
    parser.add_argument("--scenario_name", type=str, default="WVUCITY_HWFET",
                        help="name of driving cycle data")
    # CLTC_P  Standard_ChinaCity, CLTCP_WVUSUB_WVUINTER, WVUCITY_HWFET, CLTCP_WVUINTER
    parser.add_argument("--save_dir", type=str, default="./test5", help="directory in which saves training data and model")
    parser.add_argument("--log_dir", type=str, default="./logs5", help="directory in which saves logs")
    parser.add_argument("--file_v", type=str, default='v1', help="每次训练都须重新指定")
    # load learned model to train new model or evaluate
    parser.add_argument("--load_or_not", type=bool, default=True)
    parser.add_argument("--load_episode", type=int, default=397)
    parser.add_argument("--load_scenario_name", type=str, default="CLTCP_WVUINTER_w100_LR1e-03_v1")
    parser.add_argument("--load_dir", type=str, default="test2_SAC_CS_Beta")
    # evaluate
    parser.add_argument("--evaluate", type=bool, default=True)
    parser.add_argument("--evaluate_episode", type=int, default=5)
    parser.add_argument("--eva_dir", type=str, default="./eva")
    # all above
    args = parser.parse_args()
    return args
    