#! /bin/bash

cd hsp/
mkdir -p models
name=pick_reinforce

python main.py --env_name sp_pick --max_steps 80 \
--num_epochs 200 --epoch_size 100 \
--plot --plot_env $name --save models/$name.pt
