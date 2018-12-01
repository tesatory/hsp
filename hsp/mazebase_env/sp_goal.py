from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import mazebase.grid_game as gg
import mazebase.grid_item as gi
import mazebase.game_factory as gf


class Game(gg.GridGame2D):
    def __init__(self, opts):
        super(Game, self).__init__(opts)
        self.goal_loc = self.sample_reachable_loc(ensure_empty=True)
        if opts.get('fixed_goal'):
            self.goal_loc = (0,0)
        gi.add_goal(self, self.goal_loc, 0)
        self.nblocks = int(opts.get('nblocks') or 0)
        self.nwater = int(opts.get('nwater') or 0)
        gi.add_standard_items(self)
        self.agent = self.items_bytype['agent'][0]
        self.finished = False
        self.stat = dict()
        self.stat['success'] = 0

    def update(self):
        super(Game, self).update()
        location = self.goal_loc
        if (self.agent.attr['loc'][0] == location[0]
                and self.agent.attr['loc'][1] == location[1]):
            self.stat['success'] = 1
            self.finished = True

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

    def get_state(self):
        return self.agent.attr['loc']

    def set_state(self, state):
        self.move_item(self.agent, state)

    def set_state_id(self, id):
        i = 0
        for x in range(self.mapsize[0]):
            for y in range(self.mapsize[1]):
                if (x, y) == self.agent.attr['loc'] or self.is_loc_reachable((x, y)):
                    if i == id:
                        self.move_item(self.agent, (x, y))
                    i += 1
        return i
        
    def get_state_color(self):
        return [255, 0, 0]
            
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
        return vocab

    def all_actions(self, opts):
        actions = []
        actions.append('up')
        actions.append('down')
        actions.append('left')
        actions.append('right')
        actions.append('stop')
        return actions

def get_opts():
    game_opts = {}
    game_opts['step_cost'] = -.1
    game_opts['water_cost'] = -.2
    game_opts['fixed_goal'] = False
    game_opts['map_width'] = 7
    game_opts['map_height'] = 7
    game_opts['nblocks'] = 0
    game_opts['nwater'] = 0
    opts = {'game_opts': gf.opts_from_dict(game_opts)}    
    return opts
