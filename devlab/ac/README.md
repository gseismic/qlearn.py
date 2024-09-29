# QAC

## qac_mlp_td_demo.py 
是使用 MLP 实现的 Q-Actor-Critic (QAC) 算法，使用 TD 方法约束Critic网络的输出，使用Actor网络的输出作为动作选择，使用TD-error作为Actor网络的更新信号，使用MSE作为Critic网络的更新信号。