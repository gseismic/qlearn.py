
## MC-PG
本质：是用单episode-MC估计Q值

训练【策略网络actor】的结果:
因为: 策略梯度的期望 = 状态值函数v(s)的梯度
策略网络的训练导致：评委的打分越来越高
训练【评委网络critic】的结果:
用SARSA算法估计Q值，用观测奖励校准critic的打分
评委网络的训练导致：评委的打分越来越准


## 参考
https://github.com/kengz/SLM-Lab/blob/master/slm_lab/env/base_env.py
