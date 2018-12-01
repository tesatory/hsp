from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import mazebase.grid_game as gg
import mazebase.grid_item as gi
import mazebase.game_factory as gf
import mazebase.standard_grid_actions as standard_grid_actions


class Game(gg.GridGame2D):
    def __init__(self, opts):
        super(Game, self).__init__(opts)
        self.nblocks = int(opts.get('nblocks') or 0)
        self.nwater = int(opts.get('nwater') or 0)
        gi.add_standard_items(self)
        self.agent = self.items_bytype['agent'][0]
        gi.add_cycle_switch(self, self.sample_reachable_loc(), 2)
        self.switch = self.items_bytype['cycle_switch'][0]
        if opts['test_mode']:
            self.switch.setc(0)
        self.agent.replace_action('toggle_close',
                                  standard_grid_actions.toggle_close)
        self.finished = False
        self.switch_prev_color = self.switch.color
        self.reset_stat()
        
    def reset_stat(self):
        self.stat = dict()
        self.stat['success'] = 0
        self.stat['switch'] = 0

    def update(self):
        super(Game, self).update()
        if (self.switch.color != self.switch_prev_color):
            self.stat['switch'] = 1
            self.stat['success'] = 1
            self.finished = True
        self.switch_prev_color = self.switch.color

    def get_reward(self):
        r = self.opts['step_cost']
        r += self.agent.touch_cost()
        if self.finished:
            r = 1
        else:
            r = 0
        return r

    def get_state(self):
        return (self.agent.attr['loc'], self.switch.color)

    def set_state(self, state):
        l, c = state
        self.move_item(self.agent, l)
        self.switch.setc(c)
        self.switch_prev_color = self.switch.color
        
    def set_state_id(self, id):
        i = 0
        for c in range(2):
            for x in range(self.mapsize[0]):
                for y in range(self.mapsize[1]):
                    if i == id:
                        self.move_item(self.agent, (x, y))
                        self.switch.setc(c)
                    i += 1
        return i
        
    def get_state_color(self):
        if self.switch.color == 1:
            return [255, 0, 0]
        else:
            return [0, 255, 0]        

class Factory(gf.GameFactory):
    def __init__(self, game_name, game_opts, Game):
        super(Factory, self).__init__(game_name, game_opts, Game)
        ro = ('map_width', 'map_height', 'step_cost', 'nblocks', 'nwater',
              'water_cost')
        self.games[game_name]['required_opts'] = ro

    def all_vocab(self, opts):
        vocab = []
        vocab.append('corner')
        vocab.append('block')
        vocab.append('water')
        vocab.append('agent')
        vocab.append('agent0')
        vocab.append('cycle_switch')
        for s in range(2):
            vocab.append('color' + str(s))
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

def get_opts_with_args(args):
    game_opts = {}
    if args.sp:
        game_opts['test_mode'] = False
    else:
        game_opts['test_mode'] = True
    game_opts['step_cost'] = -.1
    game_opts['water_cost'] = -.2
    game_opts['map_width'] = 7
    game_opts['map_height'] = 7
    game_opts['nblocks'] = 0
    game_opts['nwater'] = 0
    opts = {'game_opts': gf.opts_from_dict(game_opts)}    
    return opts
