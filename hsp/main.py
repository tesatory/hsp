import sys
import time
import argparse
from collections import OrderedDict

import numpy as np
import torch
import data
from models import *
from utils import *
from action_utils import parse_action_args
from trainer import Trainer
from multi_threading import ThreadedTrainer
import self_play
import env_wrappers

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='Reinforce, Asymmetric Self-Play, Hiearchical Self-Play')
# training related
# note: number of steps per epoch = epoch_size X batch_size x nthreads
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of training epochs')
parser.add_argument('--epoch_size', type=int, default=10,
                    help='number of update iterations in an epoch')
parser.add_argument('--batch_size', type=int, default=500,
                    help='number of steps before each update (per thread)')
parser.add_argument('--nthreads', type=int, default=16,
                    help='How many threads to run')
# model related
parser.add_argument('--hid_size', default=64, type=int,
                    help='hidden layer size')
parser.add_argument('--recurrent', action='store_true', default=False,
                    help='make the model recurrent in time')
# optimization, Reinforce training related
parser.add_argument('--gamma', type=float, default=1.0,
                    help='discount factor between steps')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (might not work when nthreads > 0)')
parser.add_argument('--normalize_rewards', action='store_true', default=False,
                    help='normalize rewards in each batch')
parser.add_argument('--lrate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--entr', type=float, default=0,
                    help='entropy regularization coeff')
parser.add_argument('--value_coeff', type=float, default=0.05,
                    help='coeff for value loss term')
parser.add_argument('--freeze', default=False, action='store_true',
                    help='freeze the model (no learning)')
parser.add_argument('--reward_scale', type=float, default=1.0,
                    help='scale reward before backprop')
# environment related
parser.add_argument('--env_name', default="Cartpole",
                    help='name of the environment to run')
parser.add_argument('--max_steps', default=20, type=int,
                    help='force to end the game after this many steps')
parser.add_argument('--nactions', default='1', type=str,
                    help='the number of agent actions (1 for continuous). Use N:M:K for multiple actions')
parser.add_argument('--action_scale', default=1.0, type=float,
                    help='scale action output from model')
# ant task
parser.add_argument('--ant_frame_skip', default=1, type=int,
                    help='skip frames in mujoco envs (only works in some envs!)')
parser.add_argument('--ant_health_penalty', action='store_true', default=False,
                    help='penalty -10 for dying')
# other
parser.add_argument('--plot', action='store_true', default=False,
                    help='plot training progress')
parser.add_argument('--plot_env', default='main', type=str,
                    help='plot env name')
parser.add_argument('--save', default='', type=str,
                    help='save the model after training')
parser.add_argument('--load', default='', type=str,
                    help='load the model')
parser.add_argument('--hand_control', action='store_true', default=False,
                    help='hand control agent during disp')
# self-play related
parser.add_argument('--sp', default=False, action='store_true',
                    help='enable self-play')
parser.add_argument('--sp_mode', default='reverse', type=str,
                    help='self-play mode: reverse | repeat | compete')
parser.add_argument('--sp_test_rate', default=0, type=float,
                    help='percentage of target task episodes')
parser.add_argument('--sp_reward_coeff', default=0.01, type=float,
                    help='multiply rewards in self-play mode')
parser.add_argument('--sp_state_thres', default=0, type=float,
                    help='threshold of success for Bob')
parser.add_argument('--sp_persist', default=0, type=int,
                    help='start next self-play episode from previous one for K episodes')
parser.add_argument('--sp_persist_discount', default=1.0, type=float,
                    help='discount coeff between persist episodes')
parser.add_argument('--sp_persist_success', default=False, action='store_true',
                    help='only persist if prev success')
parser.add_argument('--sp_persist_separate', default=False, action='store_true',
                    help='keep Alice and Bob trajectory separate')
parser.add_argument('--sp_alice_entr', default=0.0, type=float,
                    help='entropy regularization for Alice')
parser.add_argument('--sp_extra_action', default=False, action='store_true',
                    help='add an extra action for Bob to decide stop')
parser.add_argument('--sp_asym', default=False, action='store_true',
                    help='Asymmetric self-play mode')

# learn better goal vector representation for Bob in self-play
parser.add_argument('--goal_dim', default=2, type=int,
                    help='goal representation dimension')
parser.add_argument('--goal_diff', default=False, action='store_true',
                    help='encode goal as difference (e.g. g=enc(s*)-enc(s_t))')
parser.add_argument('--sp_steps', default=5, type=int,
                    help='self-play length')
parser.add_argument('--sp_imitate', default=0, type=float,
                    help='encourage Bob to imitate Alices actions')
# Charlie
parser.add_argument('--charlie', default=False, action='store_true',
                    help='test on target task')
parser.add_argument('--goal_load', default='', type=str,
                    help='load the goal policy model only')
parser.add_argument('--charlie_action_limit', default=0.5, type=float,
                    help='limit on charlie action')
parser.add_argument('--charlie_finetune', default=0, type=float,
                    help='fine tune the lower goal policy with rewards')
parser.add_argument('--charlie_nodiff', default=False, action='store_true',
                    help='ignore goal_diff and directly feed Charlies action to the goal policy ')
args = parser.parse_args()
print(args)

if args.sp_asym:
    assert args.sp_extra_action

if args.charlie:
    d = torch.load(args.goal_load)
    target_enc = self_play.TargetEncoder(d['args'])
    goal_policy = self_play.GoalPolicy(d['args'])
    goal_policy.load_state_dict(d['goal_policy'])
    target_enc.load_state_dict(d['target_enc'])
    if args.charlie_finetune > 0:
        # train except the value head, which is not used
        goal_policy.value_head.weight.requires_grad = False
        goal_policy.value_head.bias.requires_grad = False
    for p in goal_policy.parameters():
        p.data.share_memory_()
    for p in target_enc.parameters():
        p.data.share_memory_()
    def init_env(args):
        env = data.init(args.env_name, args)
        env = self_play.GoalPolicyWrapper(args, env, goal_policy, target_enc, d['args'])
        if args.sp:
            from self_play import SelfPlayWrapper
            env = SelfPlayWrapper(args, env)
        return env
else:
    def init_env(args):
        env = data.init(args.env_name, args)
        if args.sp:
            from self_play import SelfPlayWrapper
            env = SelfPlayWrapper(args, env)
        return env


env = init_env(args)
num_inputs = env.observation_dim
args.num_actions = env.num_actions
args.dim_actions = env.dim_actions
parse_action_args(args)
if args.seed >= 0:
    torch.manual_seed(args.seed)

if args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)
if args.sp:
    args.num_inputs = num_inputs
    args.obs_dim = env.env.observation_dim # actual observation dim
    policy_net = self_play.SPModel(args)

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

if args.nthreads > 1:
    trainer = ThreadedTrainer(args, lambda: Trainer(args, policy_net, init_env(args)))
else:
    trainer = Trainer(args, policy_net, init_env(args))

disp_trainer = Trainer(args, policy_net, init_env(args))
disp_trainer.display = True
def disp():
    x = disp_trainer.get_episode()

log = OrderedDict()
log['epoch'] = LogField(list(), False, None, None)
log['num_epochs'] = LogField(list(), False, None, None)
log['num_steps_total'] = LogField(list(), False, None, None)
log['reward'] = LogField(list(), True, 'num_steps_total', 'num_episodes')
log['success'] = LogField(list(), True, 'num_steps_total', 'num_episodes')
log['value_loss'] = LogField(list(), False, 'epoch', 'num_steps')
log['action_loss'] = LogField(list(), False, 'epoch', 'num_steps')
log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
if args.continuous:
    log['action_std'] = LogField(list(), True, 'epoch', 'num_steps')
if args.env_name in ['sp_switch']:
    log['switch'] = LogField(list(), True, 'epoch', 'num_episodes')
    # log['success_switch'] = LogField(list(), True, 'epoch', 'switch')
if args.env_name in ['sp_pick']:
    log['door'] = LogField(list(), True, 'epoch', 'num_episodes')
if args.env_name in ['sp_pick']:
    log['key'] = LogField(list(), True, 'epoch', 'num_episodes')
if args.env_name in ['Ant', 'AntGather']:
    log['final_dist'] = LogField(list(), True, 'epoch', 'num_episodes')
if args.charlie:
    log['num_steps_real_total'] = LogField(list(), False, None, None)
    log['reward'] = LogField(list(), True, 'num_steps_real_total', 'num_episodes')
    log['success'] = LogField(list(), True, 'num_steps_real_total', 'num_episodes')

if args.sp:
    self_play.add_log_fields(log, args)

if args.plot:
    import progressbar
    import visdom
    vis = visdom.Visdom(env=args.plot_env)
    disp_trainer.vis = vis

def plot_goals():
    env.reset()
    Y, C = self_play.get_all_goals(args, env, policy_net.bob.target_enc)

    if args.goal_dim > 3:
        Y =pca(Y, k=3)
    vis.scatter(Y, win='goals', opts={'markercolor': C})

def plot_charlie_goals(charlie_goals):
    env.reset()
    args.sp = True
    args.goal_dim = args.dim_actions
    from self_play import SelfPlayWrapper
    sp_env = SelfPlayWrapper(args, data.init(args.env_name, args))
    Y, C = self_play.get_all_goals(args, sp_env, target_enc)
    vis.scatter(Y if args.goal_dim <=2 else pca(Y, k=3), win='goals', opts={'markercolor': C})
    args.sp = False
    ng = charlie_goals.size()[0]
    cC = np.zeros((ng, 3))
    cC[:, 0] = 200
    cC[:, 2] = 100
    C = np.concatenate([C, cC], 0)
    Y = torch.cat([Y, charlie_goals], 0)

    if args.goal_dim > 3:
        Y =pca(Y, k=3)
    vis.scatter(Y, win='charlie goals', opts={'markercolor': C})


def run(num_epochs):
    for ep in range(num_epochs):
        epoch_begin_time = time.time()
        stat = dict()
        if args.plot:
            progress = progressbar.ProgressBar(max_value=args.epoch_size).start()
        for n in range(args.epoch_size):
            if args.plot:
                progress.update(n+1)
            s = trainer.train_batch()
            merge_stat(s, stat)
        epoch_time = time.time() - epoch_begin_time
        epoch = len(log['epoch'].data) + 1
        if args.sp:
            total_test_steps = stat.get('num_steps_test', 0)
            if len(log['total_test_steps'].data) > 0:
                total_test_steps += log['total_test_steps'].data[-1]
        stat_bak = dict()
        stat_bak.update(stat)
        for k, v in log.items():
            if k == 'epoch':
                v.data.append(epoch)
            elif k == 'num_steps_total' and len(v.data) > 0:
                v.data.append(stat['num_steps'] + v.data[-1])
            elif k == 'num_steps_real_total' and len(v.data) > 0:
                v.data.append(stat['num_steps_real'] + v.data[-1])
            elif k == 'total_test_steps':
                v.data.append(total_test_steps)
            else:
                if k in stat and v.divide_by is not None and stat[v.divide_by] > 0:
                    stat[k] /= stat_bak[v.divide_by]
                v.data.append(stat.get(k, 0))
        print('Epoch {}\tReward {:.2f}\tTime {:.2f}s'.format(
            epoch, stat['reward'], epoch_time
        ))
        if args.plot:
            for k, v in log.items():
                if v.plot and len(v.data) > 0:
                    vis.line(np.asarray(v.data), np.asarray(log[v.x_axis].data[-len(v.data):]),
                    win=k, opts=dict(xlabel=v.x_axis, ylabel=k))

            pos_color = None
            if args.sp:
                if 'switch_pos' in stat:
                    switch_pos = np.asarray(stat['switch_pos'])
                    eps= 1e-7 # avoid div by 0
                    pos_max = 0.8 * switch_pos.max(0) + 0.2 * switch_pos.min(0)
                    pos_min = 0.2 * switch_pos.max(0) + 0.8 * switch_pos.min(0)
                    if (pos_max - pos_min).max() > 0:
                        N = switch_pos.shape[0]
                        pos_color = np.zeros((N, 3))
                        for i in range(N):
                            pos_color[i][0] = 255 * (stat['switch_pos'][i][0] - pos_min[0]) / (pos_max[0] - pos_min[0])
                            pos_color[i][1] = 255 * (stat['switch_pos'][i][1] - pos_min[1]) / (pos_max[1] - pos_min[1])
                            pos_color[i][2] = (255 - pos_color[i][0]/2 - pos_color[i][1]/2)
                        pos_color = np.clip(pos_color, 0, 255).astype(int)
                        vis.scatter(switch_pos[:500], win='switch_pos', opts=dict(title='switch_pos', markercolor= pos_color[:500]))
                    else:
                        vis.scatter(switch_pos[:500], win='switch_pos', opts=dict(title='switch_pos'))

                if not args.sp_asym:
                    if isinstance(env.env, env_wrappers.MazeBaseWrapper):
                        plot_goals()
                    else:
                        Y = stat['target_emb_snapshot']['data']
                        if args.goal_dim > 3:
                            Y =pca(Y, k=3)
                        if pos_color is None:
                            vis.scatter(Y[:500], win='goals')
                        else:
                            assert(Y.shape[0] == pos_color.shape[0])
                            vis.scatter(Y[:500], win='goals', opts={'markercolor': pos_color[:500]})

            if 'final_pos' in stat:
                if pos_color is None:
                    vis.scatter(np.asarray(stat['final_pos'])[:500], win='final_pos', opts=dict(title='final_pos'))
                else:
                    assert(len(stat['final_pos']) == pos_color.shape[0])
                    vis.scatter(np.asarray(stat['final_pos'])[:500], win='final_pos', opts=dict(title='final_pos', markercolor= pos_color[:500]))

            vis.save([args.plot_env])

        if args.save != '':
            save(args.save)

def save(path):
    d = dict()
    d['policy_net'] = policy_net.state_dict()
    d['log'] = log
    d['trainer'] = trainer.state_dict()
    d['args'] = args
    if args.sp and (not args.sp_asym):
        d['goal_policy'] = policy_net.bob.goal_policy.state_dict()
        d['target_enc'] = policy_net.bob.target_enc.state_dict()
    torch.save(d, path)

    if args.charlie_finetune > 0:
        d2 = dict()
        if args.sp:
            d2['args'] = env.env.args_bob
            d2['goal_policy'] = env.env.goal_policy.state_dict()
            d2['target_enc'] = env.env.target_enc.state_dict()
        else:
            d2['args'] = env.args_bob
            d2['goal_policy'] = env.goal_policy.state_dict()
            d2['target_enc'] = env.target_enc.state_dict()
        torch.save(d2, path + '.bob')

def load(path):
    d = torch.load(path)
    log.clear()
    policy_net.load_state_dict(d['policy_net'])
    log.update(d['log'])
    trainer.load_state_dict(d['trainer'])

if args.load != '':
    load(args.load)

run(args.num_epochs)

if args.save != '':
    save(args.save)

if sys.flags.interactive == 0 and args.nthreads > 1:
    trainer.quit()
    import os
    os._exit(0)
