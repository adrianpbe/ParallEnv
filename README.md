# ParallEnv

A small library that provides an RL environment class for parallelization. The core class, ParallelEnv, is inspired by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/) environments, especially its vectorized environments, but it does not work in exactly the same way. The key difference is that `ParallelEnv` allows environments and policies to run concurrently in separate processes. This maximizes the throughput of RL simulations.

# Installation

To install the library just use: `pip install parallenv`.

# Usage Example

```python
import gymnasium as gym
from parallenv import ParallEnv


if __name__ == "__main__":
    batch_size = 16
    num_envs = 32
    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(num_envs)]    
    envs = ParallEnv(
        env_fns,
        batch_size=batch_size,
        num_workers=4,
    )
    envs.reset()
    for i in range(1000):
        ids, observation, reward, terminated, truncated, info = envs.gather()
        # No more resets are required since `next_step` autoreset is set by default.
        actions = envs.action_space.sample()
        # This call is not blocking, you don't have to wait for the sub-environments
        #  to run.
        envs.step(ids, actions)
    envs.close()
```
