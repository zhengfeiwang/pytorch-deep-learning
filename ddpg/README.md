# Deep Deterministic Policy Gradient [PyTorch]

### Reference

Continuous control with deep reinforcement learning ([link to arXiv](https://arxiv.org/pdf/1509.02971.pdf))

### Environment

- PyTorch 0.4.0
- gym 0.10.5
- tensorboardX 1.2 (need TensorFlow installed)

### Usage

```bash
python main.py --help
```

use --help to see all arguments, including environments, learning rate and so on.

Default environment is Pendulum-v0, 2 fully-connected layer with 400 and 300 neurons for Actor and Critic.





