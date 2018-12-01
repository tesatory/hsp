import math
import torch
import numpy as np
from rllab.envs.mujoco.ant_env import AntEnv as AntEnvOrig

class AntEnv(AntEnvOrig):

    def __init__(self, args):
        super(AntEnv, self).__init__()
        self.frame_skip = args.ant_frame_skip
        self.sp_state_mask = torch.zeros(self.observation_space.shape[0])
        self.sp_state_mask[-3:-1] = 1.0

    def reset(self):
        self.stat = dict()
        self.stat['dist'] = 0
        return super(AntEnv, self).reset()

    def step(self, action):
        obs, reward, done, info = super(AntEnv, self).step(action)

        is_healthy = np.isfinite(self._state).all() \
            and self._state[2] >= 0.2 and self._state[2] <= 1.0
        info['is_healthy'] = is_healthy

        return obs, reward, done, info

    def get_stat(self):
        stat = {}
        stat['final_pos'] = [self.position]
        stat['final_dist'] = np.linalg.norm(np.array(self.position))
        return stat

    def get_state(self):
        return self._full_state.copy()

    def set_state(self, state):
        super(AntEnv, self).reset(init_state=state)

    @property
    def position(self):
        return (self.get_body_com("torso")[0], self.get_body_com("torso")[1])
