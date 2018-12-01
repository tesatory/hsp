import mazebase.game_factory as gf

def get_opts():
    game_opts = {}
    game_opts['step_cost'] = -.1
    game_opts['water_cost'] = -.2
    game_opts['fixed_goal'] = False
    game_opts['map_width'] = 10
    game_opts['map_height'] = 10
    game_opts['nblocks'] = 0
    game_opts['nwater'] = 0
    opts = {'game_opts': gf.opts_from_dict(game_opts)}    
    return opts
