# Hierarchical Self-Play (HSP)

This is a code for running experiments in the paper [Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1811.09083).

## Setup

### Without RLLab
First install [Anaconda](https://anaconda.org/).
Then everything else can be installed by:
```
conda env create -f environment.yml
conda activate goal_emb
```
The code is multi-threaded, so make sure each thread will only use a single CPU core:
```
export OMP_NUM_THREADS=1
```
For the plotting to work, you need to have a Visdom server running (see [this](https://github.com/facebookresearch/visdom#usage) for details).

### With RLLab
First, install [rllab](https://github.com/rll/rllab) with MuJoCo following its instruction. Next, update PyTorch and install the other dependencies:
```
conda install -c pytorch pytorch==0.3.1
pip install visdom
pip install git+git://github.com/tesatory/mazebase.git@v0.1
export OMP_NUM_THREADS=1
```

## Algorithms
The code implements three different algorithms. See `experiments/` for more experiments.

### 1. Reinforce
By default, the code will use vanilla Reinforce for training. For example, run the following to train a Reinforce agent on an simple MazeBase task (reward should reach 1 after about 1.5M steps):
```
python main.py --env_name SingleGoal --max_steps 40 --plot
```

A simple rllab task (reward should pass 400 after 30 epochs):
```
python main.py --env_name Cartpole --max_steps 100 --plot
```

A mujoco task with continuous actions that discretized into 5 bins (not sure if it can learn):
```
python main.py --env_name Ant --nactions 5 --max_steps 1000 --plot
```

### 2. Asymmetric Self-play
One of the baselines in the paper is asymmetric self-play from
[https://arxiv.org/abs/1703.05407](https://arxiv.org/abs/1703.05407). Add `--sp` flag to enable self-play:
```
python main.py --env_name SparseMountainCar --nactions 5 --max_steps 500 \
--plot --sp --sp_test_rate 0.01 --sp_mode repeat --sp_state_thres 0.2
```

### 3. Hierarchical Self-play
#### Singe Goal task
In this task, an agent is in an empty room with a randomly placed goal. The task objective is to reach that goal.

First, we train Alice and Bob through self-play:
```
python main.py --env_name sp_goal --plot --sp --sp_mode repeat \
--max_steps 8 --goal_diff --goal_dim 2 --sp_steps 3 --num_epochs 50 \
--sp_reward_coeff 0.1 --sp_imitate 0.03 --entr 0.001 --save models/goal.pt
```
By the end of training, Alice's reward should decrease to zero. Also, the learned goal embeddings should look like this

![goal embeddings](https://github.com/tesatory/rlcore/raw/task_vector_release/img/goal_emb.png)

If you want, you can also type `disp()` to see Alice and Bob playing.

Next, we train Charlie on the target task:
```
python main.py --env_name sp_goal --plot --max_steps 10 --num_epochs 20 \
 --entr 0.003 --charlie --goal_load models/goal.pt --nactions 9
```


[Optional] It is possible to train another level of self-play on top of the goal policy
```
python main.py --env_name sp_goal --plot --sp --sp_mode repeat \
--max_steps 8 --goal_diff --goal_dim 2 --sp_steps 3 --num_epochs 50 \
--sp_reward_coeff 0.1 --sp_imitate 0.03 --entr 0.001 --goal_load models/goal.pt \
--charlie --nactions 9
```

#### Switch task
In this task, there is a switch instead of a goal. Also, the agent has an extra "toggle" action that will change the switch color (but only if the agent is on the switch). The task objective is to reach the switch and change its color.

```
python -i main.py --env_name sp_switch --plot --sp --sp_mode repeat \
--max_steps 10 --goal_diff --goal_dim 3 --sp_steps 4 --num_epochs 200 \
--sp_reward_coeff 0.1 --sp_imitate 1 --save /tmp/sw
```
Here `sp_imitate` makes Bob learn from Alice's actions in a supervised way.
You should observe that the switch plot should reach around 15% but go down afterwards.
Next, train Charlie on the test task:
```
python -i main.py --env_name sp_switch --plot --sp --sp_mode repeat \
--max_steps 40 --goal_diff --goal_dim 3 --sp_steps 5 --num_epochs 100 \
--sp_reward_coeff 0.1 --sp_imitate 1 --goal_load /tmp/sw --sp_test \
--goal_discrete 9
```
Note that Charlie has discrete actions. This should give about 90% success.
