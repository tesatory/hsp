from rllab.envs.box2d.mountain_car_env import MountainCarEnv

class SparseMountainCarEnv(MountainCarEnv):    

    def compute_reward(self, action):
        yield
        yield int(self.cart.position[0] >= self.goal_cart_pos) # like VIME paper

    def get_state(self):
        return (self.cart.position[0], self.cart.linearVelocity[0])
    
    def set_state(self, state):
        initial_pos, initial_vel = state
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        self.cart.linearVelocity = (initial_vel, self.cart.linearVelocity[1])

    @property
    def position(self):
        return (self.cart.position[0], self.cart.linearVelocity[0])

