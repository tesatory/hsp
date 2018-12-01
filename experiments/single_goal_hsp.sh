#! /bin/bash

cd hsp/
mkdir -p models
name=goal_hsp

# First, train a goal-policy through self-play
python main.py --env_name sp_goal --sp --sp_mode repeat \
--max_steps 8 --goal_diff --goal_dim 2 --sp_steps 3 --num_epochs 50 \
--sp_reward_coeff 0.1 --sp_imitate 0.03 --entr 0.001 \
--plot --plot_env $name --save models/$name.pt

# Next, we train Charlie on the test task
python main.py --env_name sp_goal --max_steps 10 --num_epochs 20 \
 --entr 0.003 --charlie --goal_load models/$name.pt --nactions 9 \
 --plot --plot_env ${name}_charlie --save models/${name}_charlie.pt
