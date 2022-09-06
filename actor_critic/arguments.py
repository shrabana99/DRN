import argparse

def get_args():
    parse = argparse.ArgumentParser()

    #####  environment #####
    parse.add_argument('--env-name', type=str, default='PongNoFrameskip-v4', help='the environment name') # BreakoutNoFrameskip-v4
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--env-type', type=str, default='atari', help='the type of the environment')
    
    ##### algorithm #####
    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--entropy-coef', type=float, default=0.01, help='entropy-reg')
    parse.add_argument('--value-loss-coef', type=float, default=0.5, help='the coefficient of value loss')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='the grad clip')
    parse.add_argument('--nsteps', type=int, default=5, help='the steps to update the network')

    ##### gae #####
    parse.add_argument('--use-gae', action='store_true', help='use-gae')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')

    ##### optimizer #####
    parse.add_argument('--lr', type=float, default=7e-4, help='learning rate of the algorithm')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--alpha', type=float, default=0.99, help='the alpha coe of RMSprop')
    
    ##### log/result #####
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')
    parse.add_argument('--log-interval', type=int, default=100, help='the log interval')
    parse.add_argument('--log-dir', type=str, default='logs', help='log dir')
    
    ##### frames #####
    # parse.add_argument('--total-frames', type=int, default=10000, help='the total frames for training')
    parse.add_argument('--frame-stack-size', type=int, default=4, help='the total frames to stack')
    parse.add_argument('--total-frames', type=int, default=10000000, help='the total frames for training')


    ##### machine params #####
    parse.add_argument('--num-workers', type=int, default=8, help='the number of cpu you use')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')

    args = parse.parse_args()

    return args
