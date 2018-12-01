import math
import torch
import numpy as np
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv as AntGatherEnvOrig

class AntGatherEnv(AntGatherEnvOrig):

    def __init__(self, args):
        dying_cost = -10 if args.ant_health_penalty else 0
        super(AntGatherEnv, self).__init__(dying_cost=dying_cost)
        self.sp_state_mask = torch.zeros(self.observation_space.shape[0])
        # loc of ant
        self.sp_state_mask[122] = 1.0
        self.sp_state_mask[123] = 1.0

    def step(self, action):
        obs, reward, done, info = super(AntGatherEnv, self).step(action)
        return obs, reward, done, info

    def get_stat(self):
        stat = {}
        stat['final_pos'] = [self.position]
        stat['final_dist'] = np.linalg.norm(np.array(self.position))
        return stat

    def get_state(self):
        return self.wrapped_env._full_state.copy()

    def set_state(self, state):
        self.wrapped_env.reset(init_state=state)

    @property
    def position(self):
        return (self.wrapped_env.get_body_com("torso")[0], self.wrapped_env.get_body_com("torso")[1])
