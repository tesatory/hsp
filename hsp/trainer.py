from collections import namedtuple
from collections import deque
import random
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from action_utils import *


Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'mask', 'next_state',
                                       'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.params = [p for p in policy_net.parameters() if p.requires_grad]
        if self.args.charlie_finetune > 0:
            if self.args.sp:
                self.params.extend([p for p in env.env.goal_policy.parameters() if p.requires_grad])
            else:
                self.params.extend([p for p in env.goal_policy.parameters() if p.requires_grad])
        self.optimizer = optim.RMSprop(self.params,
            lr = args.lrate, alpha=0.97, eps=1e-6)

    def get_episode(self):
        episode = []
        state = self.env.reset()
        self.env.attr['display_mode'] = self.display
        if self.display:
            if self.args.plot:
                self.env.attr['vis'] = self.vis
            self.env.display()
        stat = dict()
        switch_t = -1
        prev_hid = Variable(torch.zeros(1, self.args.hid_size), requires_grad=False)
        for t in range(self.args.max_steps):
            misc = dict()
            if self.args.recurrent:
                action_out, value, next_hid = self.policy_net([Variable(state, requires_grad=False), prev_hid])
                prev_hid = next_hid
            elif self.args.sp:
                action_out, value, _, _ = self.policy_net(Variable(state, volatile=True))
            else:
                # do forward again in compute_grad for speed-up
                action_out, value = self.policy_net(Variable(state, volatile=True))
            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            if self.args.hand_control and self.display:
                self.hand_control(actual)
            if self.args.sp:
                misc['mind'] = self.env.current_mind
            next_state, reward, done, info = self.env.step(actual)
            stat['reward'] = stat.get('reward', 0) + reward
            misc.update(info)
            done = done or t == self.args.max_steps - 1
            mask = 0 if done else 1
            misc['episode_break'] = 0 if done else 1

            if self.args.sp and info.get('sp_switched'):
                switch_t = t
                mask = 0 # disconnect episode here
            if self.args.sp and (not self.args.sp_asym) and t == switch_t + 1 > 0:
                stat['target_emb_snapshot'] = {'merge_op': 'concat' ,
                    'data': self.policy_net.bob.target_enc.target_emb_snapshot.clone()}

            if self.args.sp and self.args.sp_persist > 0:
                misc['sp_persist_count'] = self.env.persist_count

            if self.display:
                if self.args.sp:
                    print('t={}\treward={}\tmind={}'.format(t, reward, self.env.current_mind))
                else:
                    print('t={}\treward={}'.format(t, reward))
                self.env.display()

            episode.append(Transition(state, np.array([action]), action_out, value, mask, next_state, reward, misc))
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        if hasattr(self.env, 'reward_terminal'):
            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + self.env.reward_terminal())
            stat['reward'] = stat.get('reward', 0) + self.env.reward_terminal()

        if self.args.sp and not self.env.test_mode:
            episode[switch_t] = episode[switch_t]._replace(
                reward=episode[switch_t].reward + self.env.reward_terminal_mind(1))
            if switch_t > -1:
                episode[-1] = episode[-1]._replace(
                    reward=episode[-1].reward + self.env.reward_terminal_mind(2))
            stat['reward'] = 0

        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)

        if self.display:
            print('total reward={}'.format(stat['reward']))
            if self.args.sp and not self.env.test_mode:
                print('alice reward={}\tbob reward={}'.format(stat['reward_alice'], stat['reward_bob']))

        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)
        actions = torch.from_numpy(np.concatenate(batch.action, 0))
        returns = torch.Tensor(actions.size(0),1)
        advantages = torch.Tensor(actions.size(0),1)

        if self.args.sp:
            minds = [d['mind'] for d in batch.misc]
            if self.args.sp_persist > 0:
                persist_count = [d['sp_persist_count'] for d in batch.misc]

        prev_return = 0
        prev_alice_return = 0
        prev_bob_return = 0
        rewards = self.args.reward_scale * rewards
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + self.args.gamma * prev_return * masks[i]
            if self.args.sp and self.args.sp_persist > 0:
                if minds[i] == 1:
                    # Add Alice's return from the previous episode, but only inside persist episodes.
                    if masks[i] == 0: # At Alice's last step in an episode.
                        returns[i] += self.args.sp_persist_discount * prev_alice_return
                    if persist_count[i] > 0:
                        prev_alice_return = returns[i, 0]
                    else:
                        prev_alice_return = 0

                if minds[i] == 2 and self.args.sp_persist_separate:
                    # do the same with Bob
                    if masks[i] == 0:
                        returns[i] += self.args.sp_persist_discount * prev_bob_return
                    if persist_count[i] > 0:
                        prev_bob_return = returns[i, 0]
                    else:
                        prev_bob_return = 0

            prev_return = returns[i, 0]

        if self.args.recurrent:
            # can't do batch forward.
            values = torch.cat(batch.value, dim=0)
            action_out = list(zip(*batch.action_out))
            action_out = [torch.cat(a, dim=0) for a in action_out]
        else:
            # forward again in batch for speed-up
            states = Variable(torch.cat(batch.state, dim=0), requires_grad=False)
            if self.args.sp:
                action_out, values, goal_vectors, enc_vectors = self.policy_net(states)
            else:
                action_out, values = self.policy_net(states)

        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(Variable(actions, requires_grad=False), action_means, action_log_stds, action_stds)
            stat['action_std'] = action_stds.mean(dim=1, keepdim=False).sum().data[0]
        else:
            log_p_a = action_out
            log_prob = multinomials_log_density(Variable(actions, requires_grad=False), log_p_a)
        action_loss = -Variable(advantages, requires_grad=False) * log_prob
        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.data[0]

        # value loss term
        targets = Variable(returns, requires_grad=False)
        value_loss = (values - targets).pow(2).sum()
        stat['value_loss'] = value_loss.data[0]
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.data[0]
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        if self.args.sp_alice_entr > 0:
            assert not self.args.continuous
            # entropy regularization term
            alice_entropy = 0
            alice_inds = (torch.Tensor(minds) == 1).view(-1, 1)
            if alice_inds.sum() > 0:
                for i in range(len(log_p_a)):
                    alice_log_p = log_p_a[i][alice_inds.expand_as(log_p_a[i])]
                    alice_log_p = alice_log_p.view(alice_inds.sum(), log_p_a[i].shape[1])
                    alice_entropy -= (alice_log_p * alice_log_p.exp()).sum()
                stat['alice_entropy'] = alice_entropy.data[0]
                loss -= self.args.sp_alice_entr * alice_entropy

        if self.args.sp and self.args.sp_imitate > 0:
            t = 0
            alice_sta_t = -1
            alice_end_t = -1
            train_obs = []
            train_actions = []
            while t < states.size()[0]:
                mind = states.data[t, -1]
                if mind == 1 and alice_sta_t < 0:
                        alice_sta_t = t
                elif mind == 2 and alice_sta_t >= 0 and alice_end_t < 0:
                    alice_end_t = t
                    bob_obs = states[t:t+1].data
                    alice_obs = states[alice_sta_t:alice_end_t].data.clone()
                    # augment Alice's obs so it would have correct target
                    alice_obs[:, self.args.obs_dim:] = bob_obs[:, self.args.obs_dim:].expand_as(alice_obs[:, self.args.obs_dim:])
                    train_obs.append(alice_obs)
                    train_actions.append(actions[alice_sta_t:alice_end_t])
                    alice_sta_t = -1
                    alice_end_t = -1
                t += 1
            train_obs = Variable(torch.cat(train_obs, dim=0), requires_grad=False)
            train_actions = torch.cat(train_actions, dim=0)
            bob_action_out, _, _, _ = self.policy_net(train_obs)
            assert self.args.continuous == False # not implemented yet
            if self.args.sp_extra_action:
                # no need to imitate the extra action
                train_actions = train_actions[:,:-1]
                bob_action_out = bob_action_out[:-1]
            bob_log_prob = multinomials_log_density(Variable(train_actions, requires_grad=False), bob_action_out)
            imitate_loss = -bob_log_prob.sum()
            stat['imitate_loss'] = imitate_loss.data[0]
            loss += self.args.sp_imitate * imitate_loss

        if self.args.charlie_finetune > 0:
            ft_actions = []
            ft_action_out = []
            ft_advantages = []
            for i in range(len(batch.misc)):
                ft_actions.extend(batch.misc[i]['ft_action'])
                ft_action_out.extend(batch.misc[i]['ft_action_out'])
                ft_advantages.extend([advantages[i][0] for _ in batch.misc[i]['ft_action']])

            ft_actions = torch.Tensor(np.stack(ft_actions))
            ft_action_out = list(zip(*ft_action_out))
            ft_action_out = [torch.cat(a, dim=0) for a in ft_action_out]
            ft_advantages = torch.Tensor(ft_advantages).view(-1, 1)

            env = self.env
            if self.args.sp:
                env = self.env.env
            if env.args_bob.continuous:
                ft_action_means, ft_action_log_stds, ft_action_stds = ft_action_out
                ft_log_prob = normal_log_density(Variable(ft_actions, requires_grad=False), ft_action_means, ft_action_log_stds, ft_action_stds)
            else:
                ft_log_p_a = ft_action_out
                ft_log_prob = multinomials_log_density(Variable(ft_actions, requires_grad=False), ft_log_p_a)
            ft_action_loss = -Variable(ft_advantages, requires_grad=False) * ft_log_prob
            ft_action_loss = ft_action_loss.sum()
            loss = loss + ft_action_loss

        loss.backward()
        return stat

    def run_batch(self):
        batch = []
        stat = dict()
        stat['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            episode, episode_stat = self.get_episode()
            merge_stat(episode_stat, stat)
            stat['num_episodes'] += 1
            batch += episode
            if self.args.sp and self.args.sp_persist > 0:
                # do not interrupt during persisting episodes
                while self.env.should_persist():
                    episode, episode_stat = self.get_episode()
                    merge_stat(episode_stat, stat)
                    stat['num_episodes'] += 1
                    batch += episode

        stat['num_steps'] = len(batch)
        stat['num_epochs'] = 1
        batch = Transition(*zip(*batch))
        return batch, stat

    # only used when nthreads=1
    def train_batch(self):
        batch, stat = self.run_batch()
        self.optimizer.zero_grad()
        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        if not self.args.freeze:
            self.optimizer.step()
        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def hand_control(self, actual):
        if self.args.sp:
            action_list = self.env.env.factory.actions
        else:
            action_list = self.env.factory.actions
        print(action_list)
        a = input('Enter action id (can use asdw): ')
        if a == '':
            pass
        elif a == 'w':
            actual[0] = action_list['up']
        elif a == 'a':
            actual[0] = action_list['left']
        elif a == 's':
            actual[0] = action_list['down']
        elif a == 'd':
            actual[0] = action_list['right']
        else:
            actual[0] = int(a)
