from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import random
import mazebase.grid_game as gg
import mazebase.grid_item as gi
import mazebase.game_factory as gf
import mazebase.standard_grid_actions as standard_grid_actions


class Game(gg.GridGame2D):
    def __init__(self, opts):
        super(Game, self).__init__(opts)
        self.door_x = random.randint(1, self.mapsize[0] - 2)
        self.door_y = random.randint(0, self.mapsize[1] - 1)
        self.door = gi.PickableKeyOpenedDoor(
            {'loc': (self.door_x, self.door_y)},
            key=0)
        self.add_prebuilt_item(self.door)

        # add wall
        wall_x = self.door_x
        for wall_y in range(self.mapsize[1]):
            if wall_y != self.door_y:
                gi.add_block(self, loc=(wall_x, wall_y))

        self.goal_loc = self.sample_reachable_loc(ensure_empty=True)
        gi.add_goal(self, self.goal_loc, 0)

        self.nblocks = int(opts.get('nblocks') or 0)
        self.nwater = int(opts.get('nwater') or 0)
        gi.add_standard_items(self)
        self.agent = self.items_bytype['agent'][0]

        # add switch opposite of goal
        key_y = random.randint(0, self.mapsize[1] - 1)
        if self.goal_loc[0] < wall_x:
            key_x = random.randint(wall_x + 1, self.mapsize[0] - 1)
        else:
            key_x = random.randint(0, wall_x - 1)
        self.key = gi.PickableKey({'loc': (key_x, key_y)}, 0)
        self.add_prebuilt_item(self.key)

        self.agent.replace_action('toggle_close',
            standard_grid_actions.toggle_close)

        # move agent opposite of goal
        agent_y = random.randint(0, self.mapsize[1] - 1)
        if self.goal_loc[0] < wall_x:
            agent_x = random.randint(wall_x + 1, self.mapsize[0] - 1)
        else:
            agent_x = random.randint(0, wall_x - 1)
        self.move_item(self.agent, (agent_x, agent_y))

        self.finished = False
        self.reset_stat()

    def reset_stat(self):
        self.stat = dict()
        self.stat['success'] = 0
        self.stat['door'] = 0
        self.stat['key'] = 0
        self.key_prev_state = self.is_key_picked()

    def is_key_picked(self):
        return '@picked_key' in self.agent.attr

    def is_goal_reached(self):
        location = self.goal_loc
        return (self.agent.attr['loc'][0] == location[0]
                and self.agent.attr['loc'][1] == location[1])

    def update(self):
        super(Game, self).update()
        if self.is_key_picked() != self.key_prev_state:
            self.stat['key'] = 1
        if (self.agent.attr['loc'][0] == self.door_x
                and self.agent.attr['loc'][1] == self.door_y):
            self.stat['door'] = 1
        if self.is_goal_reached():
            self.stat['success'] = 1
            self.finished = True
        self.key_prev_state = self.is_key_picked()

    def get_reward(self):
        r = self.opts['step_cost']
        r += self.agent.touch_cost()
        location = self.goal_loc
        if (self.agent.attr['loc'][0] == location[0]
                and self.agent.attr['loc'][1] == location[1]):
            r = 1
        else:
            r = 0
        return r

    def pick_key(self):
        if not self.is_key_picked():
            self.key.toggle(self.agent)
            self.door.update(self)

    def drop_key(self):
        if self.is_key_picked():
            del self.agent.attr['@picked_key']
            self.add_prebuilt_item(self.key)
            self.door.update(self)

    def get_state(self):
        return (self.agent.attr['loc'], self.is_key_picked())

    def set_state(self, state):
        l, key_picked = state
        self.move_item(self.agent, l)
        if key_picked:
            self.pick_key()
        else:
            self.drop_key()

    def set_state_id(self, id):
        i = 0
        for k in range(2):
            for x in range(self.mapsize[0]):
                for y in range(self.mapsize[1]):
                    if (x, y) == self.agent.attr['loc'] or self.is_loc_reachable((x, y)):
                        if i == id:
                            if k == 0:
                                self.pick_key()
                            else:
                                self.drop_key()
                            self.move_item(self.agent, (x, y))
                        i += 1
        return i

    def get_state_color(self):
        if self.is_key_picked():
            return [255, 0, 0]
        else:
            return [0, 255, 0]

    def oracle_step_loc(self, loc):
        x, y = self.agent.attr['loc']
        x2, y2 = loc
        possible_locs = []
        if y < y2 and self.is_loc_reachable((x, y+1)):
            possible_locs.append((x, y+1))
        if y > y2 and self.is_loc_reachable((x, y-1)):
            possible_locs.append((x, y-1))
        if x < x2 and self.is_loc_reachable((x+1, y)):
            possible_locs.append((x+1, y))
        if x > x2 and self.is_loc_reachable((x-1, y)):
            possible_locs.append((x-1, y))
        # if len(possible_locs) == 0:
        #     import pdb; pdb.set_trace()
        self.move_item(self.agent, random.choice(possible_locs))

    def oracle_step(self):
        if not self.is_key_picked():
            key_loc = self.key.attr['loc']
            if key_loc != self.agent.attr['loc']:
                self.oracle_step_loc(key_loc)
            else:
                self.pick_key()
        elif (self.goal_loc[0] < self.door_x < self.agent.attr['loc'][0]) or \
            (self.goal_loc[0] > self.door_x > self.agent.attr['loc'][0]):
            self.oracle_step_loc(self.door.attr['loc'])
        else:
            self.oracle_step_loc(self.goal_loc)


class Factory(gf.GameFactory):
    def __init__(self, game_name, game_opts, Game):
        super(Factory, self).__init__(game_name, game_opts, Game)
        ro = ('map_width', 'map_height', 'step_cost', 'nblocks', 'nwater',
              'water_cost','fixed_goal')
        self.games[game_name]['required_opts'] = ro

    def all_vocab(self, opts):
        vocab = []
        vocab.append('corner')
        vocab.append('block')
        vocab.append('water')
        vocab.append('agent')
        vocab.append('agent0')
        vocab.append('goal')
        vocab.append('goal0')
        vocab.append('pickable_key')
        vocab.append('pickable_key_opened_door')
        vocab.append('picked_key0')
        vocab.append('key0')
        return vocab

    def all_actions(self, opts):
        actions = []
        actions.append('up')
        actions.append('down')
        actions.append('left')
        actions.append('right')
        actions.append('stop')
        actions.append('toggle_close')
        return actions

def get_opts():
    game_opts = {}
    game_opts['step_cost'] = -.1
    game_opts['water_cost'] = -.2
    game_opts['map_width'] = 8
    game_opts['map_height'] = 8
    game_opts['nblocks'] = 0
    game_opts['nwater'] = 0
    opts = {'game_opts': gf.opts_from_dict(game_opts)}
    return opts
