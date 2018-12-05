#! /bin/bash

cd hsp/
mkdir -p models
name=ant_hsp

# First, train a goal-policy through self-play
python main.py --env_name Ant --nactions 5 --max_steps 120 \
--sp --sp_mode repeat --sp_state_thres 0.25 --goal_diff --goal_dim 2 \
--sp_steps 50 --sp_imitate 0.03 --num_epochs 400 --epoch_size 100 \
--sp_alice_entr 0.01 --sp_persist 3 --sp_persist_discount 0.7 \
--sp_persist_success --sp_persist_separate \
--ant_health_penalty --plot --plot_env $name --save models/$name.pt

# Next, we train Charlie on the test task
python main.py --env_name AntGather --max_steps 20 --num_epochs 300 \
--charlie --goal_load models/$name.pt --epoch_size 1 \
--charlie_action_limit 5 --action_scale 0.3 --charlie_nodiff
--charlie_finetune 0.001 --reward_scale 1 --batch_size 250 \
--ant_health_penalty --plot --plot_env ${name}_charlie \
--save models/${name}_charlie.pt
