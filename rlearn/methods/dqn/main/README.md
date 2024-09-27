 
## 注意事项
### 梯度裁剪
对稳定训练效果显著，特别是在处理复杂或不稳定的环境时。
Significantly effective in stabilizing training, especially when dealing with complex or unstable environments.

通过这些修改，我们实现了梯度裁剪，这可以帮助稳定训练过程，特别是在处理复杂或不稳定的环境时。你可以通过调整max_grad_norm的值来控制裁剪的程度。较小的值会导致更激进的裁剪，可能会稳定训练但可能会减慢学习速度；较大的值则允许更大的梯度，可能会加快学习但也可能导致不稳定。
建议在不同的环境中尝试不同的max_grad_norm值，找到最适合你的特定问题的设置。

Through these modifications, we have implemented gradient clipping, which can help stabilize the training process, particularly when handling complex or unstable environments. You can control the degree of clipping by adjusting the value of max_grad_norm. Smaller values will result in more aggressive clipping, which may stabilize training but could slow down learning; larger values allow for larger gradients, which may speed up learning but could also lead to instability.
It is recommended to try different max_grad_norm values in different environments to find the setting that best suits your specific problem.

## NoisyNet
TODO
drafts/dqn_rainbow_sonnet.py
```
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
```

## 参考文献
[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
