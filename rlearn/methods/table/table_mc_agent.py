import torch
from ...errcode import ExitCode
from ...logger import sys_logger


class TableMCAgent:
    """policy iteration Method
    """

    def __init__(self, env, verbose_freq=10, logger=None, *args, **kwargs):
        self.env = env
        self.verbose_freq = verbose_freq
        self.logger = logger or sys_logger

    def learn(self,
              initial_policy_table,
              gamma,
              num_episodes,
              initial_Q_table=None,
              episode_start_state=None,
              eps_explore=0.1,
              logger=None,
              verbose_freq=1,
              verbose: int = 1):
        """蒙特卡罗策略迭代Monto-Carlo (with eps-greedy option)

        Args:
            initial_Q_table: Pr, [n_state, n_action]
            - episode_start_state 生成episode的初始状态，为None时随机选择
            - episode_start_action 生成episode的初始action，为None时随机选择
            - eps_explore 探索量
        """
        # check input
        assert 0 < gamma <= 1.0
        assert initial_policy_table.shape == (len(env.state_space),
                                              len(env.action_space))
     
        assert 0 < eps_explore < 1
        logger = logger or sys_logger

        policy_table = initial_policy_table.clone()
        states, actions = env.observation_space, env.action_space
        
        n_state, n_action = len(states), len(actions)
        p_explore = 1/n_action * eps_explore
        p_max = 1 - (n_action - 1) * p_explore
        # print(f'{p_explore=}, {p_max=}, {p_explore * (n_action-1) + p_max=}')

        if initial_Q_table is None:
            Q_table = torch.zeros(len(states), len(actions))
        elif isinstance(initial_Q_table, str):
            assert initial_Q_table in ['zeros', 'randn', 'rand']
            Q_table = getattr(torch, initial_Q_table)(len(states), len(actions))
        else:
            assert initial_Q_table.shape == (len(env.state_space), 
                                             len(env.action_space))
            Q_table = initial_Q_table.clone()
            
        Returns = torch.zeros((n_state, n_action))
        Counts = torch.zeros((n_state, n_action))
        # G = torch.zeros(len(states), len(actions))
        
        exit_code = ExitCode.EXIT_SUCC
        i_episode = -1
        visited = defaultdict(int)

        # NOTE: [@2024-08-28 16:32:16]
        # 可能会出现两个状态来回跳动，需要超长时间找到目的状态的情况
        # 而这又导致Q-value不能被正常估计
        while True:
            # 使用最新policy获得: (state, action, reward) 
            # 可统计每个(state,action)的访问次数
            # 相当于把reward，加权时间后，放回Q_table
            # 问题：
            #   如果对于某个状态s, 路径sar_list上 
            #       gen_policy 产生了动作次数为{a0:5, a1: 1, a2: 0}
            #       a1 次数不足，利用平均做估计值可能不准
            # XXX 通过设置start_state 
            i_episode += 1
            
            # print(i_episode)
            policy = env.make_policy(policy_table)
            if episode_start_state is not None:
                start_state = episode_start_state 
            else:
                start_state = env.get_random_state()
            
            # print(f'Generating ..{start_state}')
            episode, done = env.generate_episode(policy, start_state=start_state, max_size=None)
            assert done is True
            
            # 有效的episode才加
            if (i_episode+1) % verbose_freq == 0:
                logger.info(f'{i_episode+1}/{num_episodes}: episode length: {len(episode)}')

            # print(f'{episode=}')
            g = 0

            sa_episode = [(item[0], item[1]) for item in episode]
            # print(sa_episode)
            for ii, (state, action, reward) in enumerate(episode):
                g = gamma * g + reward
                # 该方法意味着我们只考虑每个状态-动作对在一个回合中第一次出现时来计算回报
                if (state, action) not in sa_episode[:ii]:
                    i_state = env.index_of_state(state)
                    i_action = env.index_of_action(action)
                    Returns[i_state, i_action] += g
                    Counts[i_state, i_action] += 1
                    visited[(i_state, i_action)] += 1
                    Q_table[i_state, i_action] = Returns[i_state, i_action]/Counts[i_state, i_action]
                else:
                    # print(f'skip: {state, action=}')
                    pass
            # for (i_state, i_action) in visited:
            #     # 策略估计
            #     p = Returns[i_state, i_action]/Counts[i_state, i_action]
            #     Q_table[i_state, i_action] = p
                # print(f'{reward=}, {Returns[i_state, i_action]=}, {Counts[i_state, i_action]=} {state, action=}: {p=}')
            
            # 策略改进
            # print('update policy ...')
            _, max_indices = torch.max(Q_table, dim=1)
            if i_episode >= num_episodes:
                policy_table[:, :] = 0
                policy_table[torch.arange(0, len(states)), max_indices] = 1
                if verbose >= 0:
                    logger.info('Exit Reach num_episodes')
                break
            else:
                policy_table[:, :] = p_explore
                policy_table[torch.arange(0, len(states)), max_indices] = p_max
            
        info = {
            'visited': visited
        }
        state_values = None
        return exit_code, policy_table, (Q_table, state_values), info
