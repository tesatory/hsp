#! /bin/bash

cd hsp/
mkdir -p models
name=ant_sp

python main.py --env_name AntGather --max_steps 1000 --nactions 5 \
--sp --sp_asym --sp_mode repeat --sp_extra_action --sp_state_thres 0.2 \
--sp_test_rate 0.1 --ant_health_penalty --sp_alice_entr 0.003 \
--sp_reward_coeff 0.01 --num_epochs 300 --epoch_size 100 \
--plot --plot_env $name --save models/$name.pt
