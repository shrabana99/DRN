import os
from arguments import get_args
from envs import create_env
from a2c_agent import a2c_agent
from log_utils import Log


if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # get args
    args = get_args()
    # get logs
    log = Log(args)
    # create environments
    envs, single_action_space = create_env(args)
    print("1....................", envs.observation_space) #  (84, 84, 4)
    # set seeds
    # set_seeds(args)
    # create trainer
    a2c_trainer = a2c_agent(envs, single_action_space, log, args)
    a2c_trainer.learn()
    
    # close the environment
    envs.close()
