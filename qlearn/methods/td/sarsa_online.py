
"""
Initialize Q(s, a) arbitrarily for all states s and actions a
For each episode:
    Initialize state s
    Choose action a from state s using policy derived from Q (e.g., epsilon-greedy)
    For each step of the episode:
        Take action a, observe reward r, and next state s'
        Choose next action a' from state s' using policy derived from Q
        Compute the TD error: δ = r + γ * Q(s', a') - Q(s, a)
        Update Q-value: Q(s, a) = Q(s, a) + α * δ
        Set s = s', a = a'
        If s is terminal:
            Break
"""
