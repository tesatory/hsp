import sys
from env_wrappers import *

def init(env_name, args):
    if env_name == 'SparseMountainCar':
        from rllab_env.sparse_mountain_car import SparseMountainCarEnv
        env = RLLabWrapper(SparseMountainCarEnv())
    elif env_name == 'Ant':
        from rllab_env.ant_env import AntEnv
        env = RLLabWrapper(AntEnv(args))
    elif env_name == 'AntGather':
        from rllab_env.ant_gather_env import AntGatherEnv
        env = RLLabWrapper(AntGatherEnv(args))
    elif env_name == 'HalfCheetah':
        from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = RLLabWrapper(HalfCheetahEnv())
    elif env_name == 'MountainCar':
        from rllab.envs.box2d.mountain_car_env import MountainCarEnv
        env = RLLabWrapper(MountainCarEnv())
    elif env_name == 'Cartpole':
        from rllab.envs.box2d.cartpole_env import CartpoleEnv
        env = RLLabWrapper(CartpoleEnv())
    elif env_name == 'SingleGoal':
        from mazebase import single_goal
        from mazebase_env import single_goal as config
        env = MazeBaseWrapper('SingleGoal', single_goal, config)
    elif env_name == 'sp_goal':
        from mazebase_env import sp_goal
        env = MazeBaseWrapper('sp_goal', sp_goal, sp_goal)
    elif env_name == 'sp_switch':
        from mazebase_env import sp_switch
        config = sp_switch.get_opts_with_args(args)
        sp_switch.get_opts = lambda: config
        env = MazeBaseWrapper('sp_switch', sp_switch, sp_switch)
    elif env_name == 'sp_pick':
        from mazebase_env import sp_pick
        env = MazeBaseWrapper('sp_pick', sp_pick, sp_pick)
    else:
        raise RuntimeError("wrong env name")

    return env
