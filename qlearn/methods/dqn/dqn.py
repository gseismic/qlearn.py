
"""
```
Initialize Q-network with random weights
Initialize target Q-network with the same weights as Q-network
Initialize experience replay buffer

for each episode:
    Initialize state s
    for each step in the episode:
        with probability ε:
            Select a random action a
        otherwise:
            Select action a = argmax_a Q(s, a)  # Greedy action selection

        Execute action a, observe reward r and next state s'

        Store experience (s, a, r, s') in experience replay buffer

        Sample a random minibatch of experiences from the replay buffer

        for each experience (s, a, r, s') in minibatch:
            Compute target value:
                target = r + γ * max_a' Q_target(s', a')
            Compute loss:
                loss = (Q(s, a) - target)^2

        Perform gradient descent on the loss to update Q-network weights

        Every C steps:
            Update target Q-network weights with Q-network weights
```
"""
