#! /bin/bash

cd hsp/
mkdir -p models
name=ant_reinforce

python main.py --env_name AntGather --max_steps 1000 --num_epochs 500 \
--nactions 5 --ant_health_penalty \
--plot --plot_env $name --save models/$name.pt
