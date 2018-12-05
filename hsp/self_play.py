import torch.nn as nn
from torch.autograd import Variable
from utils import *
from models import MLP
import random
from argparse import Namespace
import action_utils
from env_wrappers import *


# encode the target observation into a goal vector
class TargetEncoder(nn.Module):
    def __init__(self, args):
        super(TargetEncoder, self).__init__()
        self.args = args
        self.enc = nn.Sequential(
            nn.Linear(args.obs_dim, args.hid_size),
            nn.Tanh(),
            nn.Linear(args.hid_size, args.goal_dim))

    def forward(self, x):
        obs_target, obs_current = x
        h_goal = self.enc(obs_target)
        h_current = self.enc(obs_current)
        self.target_emb_snapshot = h_goal.data # used for plotting only
        if self.args.goal_diff:
            h_goal = h_goal - h_current
        return h_goal, h_current

# takes a goal vector as input
class GoalPolicy(nn.Module):
    def __init__(self, args):
        super(GoalPolicy, self).__init__()
        self.args = args
        # current state encoder
        self.obs_enc = nn.Sequential(
            nn.Linear(args.obs_dim, args.hid_size),
            nn.Tanh(),
            nn.Linear(args.hid_size, args.hid_size))

        self.goal_enc = nn.Sequential(
            nn.Linear(args.goal_dim, args.hid_size))

        self.final_hid = nn.Sequential(
            nn.Tanh(),
            nn.Linear(args.hid_size, args.hid_size),
            nn.Tanh())

        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(args.hid_size, 1)


    def forward(self, x):
        obs_current, goal_vector = x

        # encode current state
        h_current = self.obs_enc(obs_current)
        h_goal = self.goal_enc(goal_vector)
        h_final = self.final_hid(h_current + h_goal)
        v = self.value_head(h_final)

        if self.continuous:
            action_mean = self.action_mean(h_final)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            return (action_mean, action_log_std, action_std), v
        else:
            return [F.log_softmax(head(h_final), dim=-1) for head in self.heads], v

class Bob(nn.Module):
    def __init__(self, args):
        super(Bob, self).__init__()
        self.args = args
        self.target_enc = TargetEncoder(args)
        self.goal_policy = GoalPolicy(args)

    def forward(self, x):
        obs_current = x[:, :self.args.obs_dim]
        obs_current_meta = x[:, :self.args.obs_dim+2]
        N = obs_current_meta.size()[1]
        obs_target = x[:, N:N+self.args.obs_dim]
        N += obs_target.size()[1]
        goal_vector, enc_vector = self.target_enc([obs_target, obs_current])
        a, v = self.goal_policy([obs_current, goal_vector])
        return a, v, goal_vector, enc_vector

class Alice(nn.Module):
    def __init__(self, args, is_bob=False):
        super(Alice, self).__init__()
        self.args = args
        self.is_bob = is_bob
        self.enc_obs_curr = nn.Linear(args.obs_dim, args.hid_size)
        self.enc_obs_init = nn.Linear(args.obs_dim, args.hid_size)
        self.enc_meta = nn.Linear(2, args.hid_size)
        self.affine2 = nn.Linear(args.hid_size, args.hid_size)
        self.continuous = args.continuous
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
        self.value_head = nn.Linear(args.hid_size, 1)

    def forward(self, x):
        obs_curr = x[:, :self.args.obs_dim]
        obs_meta = x[:, self.args.obs_dim:self.args.obs_dim+2]
        obs_init = x[:, self.args.obs_dim+2:self.args.obs_dim*2+2]

        h1 = self.enc_obs_curr(obs_curr)
        h1 = h1 + self.enc_obs_init(obs_init)
        h1 = h1 + self.enc_meta(obs_meta)
        h1 = F.tanh(h1)
        h2 = F.tanh(self.affine2(h1))
        v = self.value_head(h2)
        if self.continuous:
            action_mean = self.action_mean(h2)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            if self.is_bob:
                return (action_mean, action_log_std, action_std), v, None, None
            else:
                return (action_mean, action_log_std, action_std), v
        else:
            if self.is_bob:
                return [F.log_softmax(head(h2), dim=-1) for head in self.heads], v, None, None
            else:
                return [F.log_softmax(head(h2), dim=-1) for head in self.heads], v

class SPModel(nn.Module):
    def __init__(self, args):
        super(SPModel, self).__init__()
        self.args = args
        self.alice = Alice(args)

        if self.args.sp_asym:
            # Alice and Bob use the same architecture (bit hacky)
            self.bob = Alice(args, is_bob=True)
        else:
            self.bob = Bob(args)

    def forward(self, x):
        # the last element of input must be mind index
        mind = x[:,-1:].contiguous()

        if x.size()[0] > 1:
            amask = (mind == 1)
            bmask = (mind == 2)
            if amask.data.sum() == 0:
                return self.bob(x)
            if bmask.data.sum() == 0:
                y, v = self.alice(x)
                return y, v, None, None

            ax = x[amask.expand(x.size())].view(int(amask.data.sum()), x.size(1))
            bx = x[bmask.expand(x.size())].view(int(bmask.data.sum()), x.size(1))
            ay, av = self.alice(ax)
            by, bv, gv, ev = self.bob(bx)
            y = [Variable(mind.data.new(x.size(0), it.size(1))) for it in ay]
            for i in range(len(ay)):
                y[i].masked_scatter_(amask.expand(y[i].size()), ay[i])
                y[i].masked_scatter_(bmask.expand(y[i].size()), by[i])
            v = mind.clone()
            v.masked_scatter_(amask, av)
            v.masked_scatter_(bmask, bv)

            return y, v, gv, ev
        elif mind.data[0][0] == 1:
            y, v = self.alice(x)
            return y, v, None, None
        elif mind.data[0][0] == 2:
            return self.bob(x)
        else:
            raise RuntimeError("wtf")


class SelfPlayWrapper(EnvWrapper):
    def __init__(self, args, env):
        super(SelfPlayWrapper, self).__init__(env)
        assert args.sp
        self.args = args
        self.total_steps = 0
        self.total_test_steps = 0
        self.attr['display_mode'] = False
        self.persist_count = self.args.sp_persist

    @property
    def observation_dim(self):
        dim = self.env.observation_dim # current observation
        dim += 2 # meta information
        dim += self.env.observation_dim # target observation
        dim += 1 # current mind
        return dim

    def should_persist(self):
        should_persist = True
        if self.args.sp_persist <= self.persist_count:
            # above limit
            should_persist = False
        if should_persist and self.alice_last_state is None:
            should_persist = False
        if should_persist and self.args.sp_persist_success and self.success == False:
            # prev episode wasn't success
            should_persist = False
        if should_persist and self.args.sp_persist_separate and self.bob_last_state is None:
            should_persist = False
        return should_persist

    def reset(self):
        self.stat = dict()
        self.stat['reward_test'] = 0
        self.stat['reward_alice'] = 0
        self.stat['reward_bob'] = 0
        self.stat['num_steps_test'] = 0
        self.stat['num_steps_alice'] = 0
        self.stat['num_steps_bob'] = 0
        self.stat['num_episodes_test'] = 0
        self.stat['num_episodes_alice'] = 0
        self.stat['num_episodes_bob'] = 0


        if self.should_persist():
            self.env.set_state(self.alice_last_state)
            self.env.reset_stat()
            self.persist_count += 1
        else:
            self.env.reset()
            self.persist_count = 0
        self.alice_last_state = None
        self.initial_state = self.env.get_state() # bypass wrapper
        self.initial_pos = self.get_pos()
        self.current_time = 0
        self.current_mind_time = 0
        self.success = False
        self.display_obs = []
        self.target_obs_curr = []
        self.target_reached_curr = 0
        if self.total_test_steps < self.args.sp_test_rate * self.total_steps:
            assert self.args.sp_asym
            self.test_mode = True
            self.current_mind = 2
            self.target_obs = torch.zeros((1, self.env.observation_dim))
        else:
            self.target_obs = self.env.get_current_obs()
            self.current_mind = 1
            self.test_mode = False
            if self.args.env_name == 'AntGather':
                # no apple during self-play
                self.env.env.objects = []

        return self.get_current_obs()

    def get_current_obs(self):
        current_obs = self.env.get_current_obs()
        if self.test_mode:
            mode = 1
            time = 0
        else:
            mode = -1
            time = self.current_time / self.args.max_steps
        obs = current_obs
        obs = torch.cat((obs, torch.Tensor([[mode, time]])), dim=1)
        obs = torch.cat((obs, self.target_obs), dim=1)
        obs = torch.cat((obs, torch.Tensor([[self.current_mind]])), dim=1)
        return obs

    def get_bob_diff(self, obs, target):
        sp_state_mask = self.env.property_recursive('sp_state_mask')
        if not sp_state_mask is None:
            diff = torch.dist(target.view(-1) * sp_state_mask,
                        obs.view(-1) * sp_state_mask)
        else:
            diff = torch.dist(target, obs)
        return diff


    def step(self, action):
        if self.args.sp_extra_action:
            extra_action = action[-1]
            action = action[:-1]

        self.current_time += 1
        self.current_mind_time += 1
        obs_internal, reward, done, info = self.env.step(action)
        self.total_steps += 1
        if self.test_mode:
            self.total_test_steps += 1
            self.stat['num_steps_test'] += 1
            self.stat['reward_test'] += reward
            return (self.get_current_obs(), reward, done, info)
        done = False
        reward = 0
        if 'is_healthy' in info and info['is_healthy'] == False:
            if self.args.ant_health_penalty:
                reward = -10
            done = True

        if self.current_mind == 1:
            self.stat['num_steps_alice'] += 1
            self.stat['reward_alice'] += reward

            if not done:
                should_switch = False
                if self.args.sp_asym:
                    if extra_action == 1 and (not done):
                        should_switch = True
                else:
                    if self.current_time == self.args.sp_steps:
                        should_switch = True

                if should_switch:
                    if self.attr['display_mode']:
                        self.display()
                    self.switch_mind()
                    info['sp_switched'] = True
        else:
            self.stat['num_steps_bob'] += 1
            self.bob_last_state = self.env.get_state()

            if not done:
                if self.args.sp_asym:
                    reward = -self.args.sp_reward_coeff

                # check if Bob succeeded
                diff = self.get_bob_diff(obs_internal, self.target_obs)
                if bool(diff <= self.args.sp_state_thres):
                    self.success = True
                    done = True

            self.stat['reward_bob'] += reward

        obs = self.get_current_obs()
        return (obs, reward, done, info)

    def get_pos(self):
        return self.env.property_recursive('position', (0, 0))

    def switch_mind(self):
        self.current_mind_time = 0
        self.alice_last_state = self.env.get_state()
        self.current_mind = 2
        self.stat['switch_time'] = self.current_time
        self.stat['switch_pos'] = [self.get_pos()]
        self.stat['switch_dist'] = np.linalg.norm(np.array(self.get_pos()) - np.array(self.initial_pos))
        if self.args.sp_mode == 'reverse':
            pass
        elif self.args.sp_mode == 'repeat':
            self.target_obs = self.env.get_current_obs()
            if self.persist_count > 0 and self.args.sp_persist_separate:
                # start Bob from his previous state
                self.env.set_state(self.bob_last_state) # bypass wrapper
            else:
                self.env.set_state(self.initial_state) # bypass wrapper
        else:
            raise RuntimeError("wtf")
        self.bob_initial_pos = self.get_pos()
        if self.attr['display_mode']:
            input()

    def reward_terminal(self):
        if self.test_mode and hasattr(self.env, 'reward_terminal'):
            return self.env.reward_terminal()
        else:
            return 0

    def reward_terminal_mind(self, mind):
        if self.test_mode:
            return 0

        if mind == 1:
            if self.current_mind == 2:
                if self.args.sp_asym:
                    ta = self.stat['switch_time']
                    tb = self.current_time - self.stat['switch_time']
                    return self.args.sp_reward_coeff * max(0, (tb - ta))

                if self.success:
                    return 0
                else:
                    return 1
            else:
                return 0
        else:
            if self.args.sp_asym:
                return 0
            if self.success:
                return 1
            else:
                return 0

    def get_stat(self):
        if self.test_mode:
            self.stat['reward_test'] += self.reward_terminal()
            self.stat['num_episodes_test'] = 1
            # self.stat['test_pos'] = [self.get_pos()]
        else:
            self.stat['reward_alice'] += self.reward_terminal_mind(1)
            self.stat['num_episodes_alice'] = 1

            if self.current_mind > 1:
                self.stat['reward_bob'] += self.reward_terminal_mind(2)
                self.stat['num_episodes_bob'] = 1
                self.stat['success_bob'] = 1 if self.success else 0

            if self.stat['num_steps_bob'] > 0:
                self.stat['final_pos'] = [self.get_pos()]
                self.stat['bob_dist'] = np.linalg.norm(np.array(self.get_pos()) - np.array(self.bob_initial_pos))
            else:
                if 'switch_pos' in self.stat:
                    del self.stat['switch_pos']

            if self.args.sp_persist > 0:
                self.stat['persist_count'] = self.persist_count

        if hasattr(self.env, 'get_stat'):
            s = self.env.get_stat()
            if 'final_pos' in s:
                del s['final_pos']
            if 'success' in s:
                if self.test_mode:
                    self.stat['success_test'] = s['success']
                del s['success']
            merge_stat(s, self.stat)

        return self.stat

    def display(self):
        maze_env = self.env
        if self.args.charlie:
            maze_env = maze_env.env
        if isinstance(maze_env, MazeBaseWrapper):
            if self.current_mind == 1:
                maze_env.env.agent.attr['_display_symbol'] = (u' A ', 'blue', 'on_white', None)
            else:
                maze_env.env.agent.attr['_display_symbol'] = (u' A ', 'red', 'on_white', None)
                print('mind={} success={}'.format(self.current_mind, self.success))
        else:
            obs = self.env.get_current_obs()
            self.display_obs.append(obs)
            if len(self.display_obs) > 1:
                X = torch.cat(self.display_obs, dim=0)
                self.attr['vis'].line(X, win='sp_obs_line')
        self.env.display()

# control an environment through a goal policy
class GoalPolicyWrapper(EnvWrapper):
    def __init__(self, args, env, goal_policy, target_enc, args_bob):
        super(GoalPolicyWrapper, self).__init__(env)
        self.args = args
        self.goal_policy = goal_policy
        self.target_enc = target_enc
        self.args_bob = args_bob
        if self.args.charlie_nodiff:
            self.args_bob.goal_diff = False

    @property
    def num_actions(self):
        return 0

    @property
    def dim_actions(self):
        return self.args_bob.goal_dim

    @property
    def action_space(self):
        return Namespace(
            low=[-self.args.charlie_action_limit for _ in range(self.dim_actions)],
            high=[self.args.charlie_action_limit for _ in range(self.dim_actions)])

    def reset(self):
        self.t = 0
        self.stat = dict()
        self.stat['num_steps_real'] = 0
        return self.env.reset()

    def step(self, goal):
        self.t += 1
        info = dict()

        goal = Variable(torch.Tensor(goal).view(1, -1), volatile=True)

        cg = goal.data # for plotting purposes
        if self.args_bob.goal_diff:
            obs = Variable(self.get_current_obs(), volatile=True)
            goal = goal + self.target_enc.enc(obs)

        if self.attr['display_mode'] and (not self.args.env_name in ['AntGather']):
            obs = Variable(self.get_current_obs(), volatile=True)
            obs_emb = self.target_enc.enc(obs)
            Y = torch.cat([goal.data, obs_emb.data], dim=0)
            C = np.concatenate((np.array([[255, 255, 0]]), np.array([[0, 255, 255]])), 0)
            if isinstance(self.env, MazeBaseWrapper):
                Y2, C2 = get_all_goals(self.args_bob, self, self.target_enc)
                Y = torch.cat([Y2, Y], dim=0)
                C = np.concatenate((C2, C) ,0)
            if Y.shape[1] > 3:
                Y =pca(Y, k=3)
            self.attr['vis'].scatter(Y, win='goals', opts={'markercolor': C})
            input()

        reward_sum = 0
        if self.args.charlie_finetune > 0:
            # stop grad here during fine tuning
            goal = Variable(goal.data, requires_grad=False)
            info['ft_action'] = []
            info['ft_action_out'] = []

        for t in range(self.args_bob.sp_steps):
            obs = self.get_current_obs()
            if self.args.env_name in ['AntGather']:
                obs = obs[:,:125]

            if self.args.charlie_finetune > 0:
                obs = Variable(obs, requires_grad=False)
            else:
                obs = Variable(obs, volatile=True)
            if self.args_bob.goal_diff:
                goal_t = goal - self.target_enc.enc(obs)
            else:
                goal_t = goal
            action_out, _ = self.goal_policy([obs, goal_t])
            action = action_utils.select_action(self.args_bob, action_out)
            action, actual = action_utils.translate_action(self.args_bob, self.env, action)

            if self.args.charlie_finetune > 0:
                info['ft_action'].append(action)
                info['ft_action_out'].append(action_out)

            next_state, reward, done, info_sub = self.env.step(actual)
            reward_sum += reward
            if 'goal_distance' in info_sub:
                self.stat['goal_distance'] = info_sub['goal_distance']
            self.stat['num_steps_real'] += 1
            if self.attr['display_mode']:
                self.env.display()
            if done:
                break

        return (next_state, reward_sum, done, info)

    def display(self):
        self.env.display()


def get_all_goals(args, sp_env, target_enc):
    maze_env = sp_env.env
    N = maze_env.set_state_id(-1)
    if N is None:
        return
    orig_state = maze_env.get_state()
    goal_enc = target_enc.enc
    Y = torch.zeros(N, args.goal_dim)
    C = np.zeros((N, 3))
    for i in range(N):
        maze_env.set_state_id(i)
        x = Variable(maze_env.get_current_obs(), volatile=True)
        y = goal_enc(x)
        Y[i] = y.data
        C[i] = maze_env.get_state_color()
        if args.env_name == 'sp_gather' or args.env_name == 'sp_gather_easy' or args.env_name == 'sp_gather1_easy':
            maze_env.set_state(orig_state)
    maze_env.set_state(orig_state)
    return Y, C

def add_log_fields(log, args):
    del log['reward']
    del log['success']
    if args.sp_asym:
        log['reward_test'] = LogField(list(), True, 'total_test_steps', 'num_episodes_test')
        log['success_test'] = LogField(list(), True, 'total_test_steps', 'num_episodes_test')
    log['reward_alice'] = LogField(list(), True, 'epoch', 'num_episodes_alice')
    log['reward_bob'] = LogField(list(), True, 'epoch', 'num_episodes_bob')
    log['success_bob'] = LogField(list(), True, 'epoch', 'num_episodes_bob')
    log['num_steps_alice'] = LogField(list(), args.sp_asym, 'epoch', 'num_episodes_alice')
    log['num_steps_bob'] = LogField(list(), args.sp_asym, 'epoch', 'num_episodes_bob')
    log['total_test_steps'] = LogField(list(), False, 'epoch', None)
    if args.env_name in ['Ant', 'AntGather']:
        log['switch_dist'] = LogField(list(), True, 'epoch', 'num_episodes_bob')
        log['bob_dist'] = LogField(list(), True, 'epoch', 'num_episodes_bob')
    if args.sp_imitate > 0:
        log['imitate_loss'] = LogField(list(), True, 'epoch', 'num_steps_alice')
    if args.sp_alice_entr > 0:
        log['alice_entropy'] = LogField(list(), True, 'epoch', 'num_steps_alice')
    if args.sp_persist > 0:
        log['persist_count'] = LogField(list(), True, 'epoch', 'num_episodes_alice')
