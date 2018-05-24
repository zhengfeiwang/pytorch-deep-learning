import numpy as np
import torch


class Noise(object):
    def __init__(self, size):
        self.size = size

    def sample(self):
        return np.random.normal(size=self.size)


def to_numpy(tensor):
    return tensor.to(torch.device("cpu")).detach().numpy()


def to_tensor(ndarray, requires_grad=False, dtype=torch.float, device=torch.device("cpu")):
    tensor = torch.from_numpy(ndarray).to(device, dtype)
    return tensor.requires_grad_() if requires_grad else tensor


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - tau) + source_param * tau)


def hard_update(target, source):
    soft_update(target, source, 1.0)
