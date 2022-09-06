
import gym
from  supersuit import max_observation_v0, frame_skip_v0, resize_v1, frame_stack_v2,\
                         color_reduction_v0, gym_vec_env_v0


def create_env(args):
    env = gym.make(args.env_name)
    action_space = env.action_space

    env = max_observation_v0(env, memory=2)
    env = frame_skip_v0(env, 4)

    # env = normalize_obs_v0(env, env_min=0, env_max=1)
    env = resize_v1(env, x_size=84, y_size=84, linear_interp=True)
    env = color_reduction_v0(env, 'full')
    env = frame_stack_v2(env, args.frame_stack_size, stack_dim=0)        
    if args.num_workers == 1:
        return env, action_space
    env = gym_vec_env_v0(env, args.num_workers, multiprocessing=True)
    return env, action_space   

