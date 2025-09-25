# ParallEnv

A small library that provides an RL environment class for parallelization. The core class, ParallelEnv, is inspired by [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/) environments, especially its vectorized environments, but it does not work in exactly the same way. The key difference is that `ParallelEnv` allows environments and policies to run concurrently in separate processes. This maximizes the throughput of RL simulations.

