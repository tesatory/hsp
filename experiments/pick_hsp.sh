#! /bin/bash

cd hsp/
mkdir -p models
name=pick_hsp

# First, train a goal-policy through self-play
python main.py --env_name sp_pick --sp --sp_mode repeat \
--max_steps 13 --goal_dim 3 --sp_steps 5 --num_epochs 1000 \
--sp_imitate 0.03 --entr 0.01 --sp_persist 3 \
--sp_persist_discount 0.7 --plot --plot_env $name --save models/$name.pt

# Next, we train Charlie on the test task
python main.py  --env_name sp_pick --max_steps 8 --num_epochs 300 \
--charlie  --action_scale 0.3 --charlie_finetune 0.1 --batch_size 250 \
--epoch_size 20 --charlie_nodiff  --charlie_action_limit 5 \
--goal_load models/$name.pt --plot --plot_env ${name}_charlie \
--save models/${name}_charlie.pt
