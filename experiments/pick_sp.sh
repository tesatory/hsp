#! /bin/bash

cd hsp/
mkdir -p models
name=pick_sp

python main.py --env_name sp_pick --sp --sp_asym \
--sp_mode repeat --max_steps 50 --num_epochs 5000 \
--sp_alice_entr 0.003 --sp_extra_action \
--sp_reward_coeff 0.1 --sp_test_rate 0.1 \
--plot --plot_env $name --save models/$name.pt
