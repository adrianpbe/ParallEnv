import functools
import pytest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from parallenv.parallenv import (
    ParallEnv,
    AlreadyPendingEnvError,
    ClosedEnvError,
    AutoresetMode,
)


class IdEnv(gym.Env):
    def __init__(self, id: int):
        self.id = id
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        return np.array([0.0], dtype=np.float32), 0.0, False, False, {"env_id": self.id}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed, options=options)
        return np.array([0.0], dtype=np.float32), {"env_id": self.id}


class TerminateOnFirstStepEnv(gym.Env):
    """Environment that terminates on the first step after a reset."""

    def __init__(self):
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def step(self, action):
        return (
            np.array([1.0], dtype=np.float32),
            1.0,
            True,
            True,
            {},
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return np.array([0.0], dtype=np.float32), {"reset": True}


def test_ids_experience_alignment():
    env_fns = [functools.partial(IdEnv, i) for i in range(4)]

    envs = ParallEnv(
        env_fns=env_fns,
        batch_size=4,
        num_workers=4,
    )
    envs.reset()
    ids, observation, reward, terminated, truncated, info = envs.gather(timeout=2)
    envs.close()
    assert np.array_equal(ids, info["env_id"])


def test_already_pending_env_error_on_reset_and_step():
    envs = ParallEnv(
        env_fns=[functools.partial(IdEnv, 0), functools.partial(IdEnv, 1)],
        batch_size=2,
        num_workers=1,
    )

    # First reset sets all envs to WAITING_RESET (pending) until gathered
    envs.reset()

    with pytest.raises(AlreadyPendingEnvError):
        envs.reset()
    with pytest.raises(AlreadyPendingEnvError):
        envs.step(env_ids=[0], actions=np.array([[0.0], [0.0]], dtype=np.float32))

    _ = envs.gather()

    envs.step(env_ids=[0, 1], actions=np.array([[0.0], [0.0]], dtype=np.float32))

    with pytest.raises(AlreadyPendingEnvError):
        envs.step(env_ids=[0, 1], actions=np.array([[0.0], [0.0]], dtype=np.float32))
    with pytest.raises(AlreadyPendingEnvError):
        envs.reset()

    envs.gather()
    envs.close()


def test_closed_env_error_after_close():
    envs = ParallEnv(
        env_fns=[functools.partial(IdEnv, 0)],
        batch_size=1,
        num_workers=1,
    )
    envs.reset()
    envs.close()

    with pytest.raises(ClosedEnvError):
        envs.reset()
    with pytest.raises(ClosedEnvError):
        envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    with pytest.raises(ClosedEnvError):
        envs.gather(timeout=1)


def test_autoreset_next_step():
    envs = ParallEnv(
        env_fns=[TerminateOnFirstStepEnv],
        batch_size=1,
        num_workers=1,
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    envs.reset()
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert not (terminated or truncated)

    envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert terminated or truncated
    assert np.array_equal(np.array([[1.0]], dtype=np.float32), observation)

    envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    ids, observation, reward, terminated, truncated, info = envs.gather()
    envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    assert "reset" in info
    assert not (terminated or truncated)


def test_autoreset_same_step():
    envs = ParallEnv(
        env_fns=[TerminateOnFirstStepEnv],
        batch_size=1,
        num_workers=1,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    envs.reset()
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert not (terminated or truncated)
    envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert terminated or truncated
    assert "final_obs" in info
    assert np.array_equal(np.array([1.0], dtype=np.float32), info["final_obs"][0])
    assert "final_info" in info
    assert np.array_equal(np.array([[0.0]], dtype=np.float32), observation)
    assert "reset" in info


def test_autoreset_disable():
    envs = ParallEnv(
        env_fns=[TerminateOnFirstStepEnv],
        batch_size=1,
        num_workers=1,
        autoreset_mode=AutoresetMode.DISABLED,
    )

    envs.reset()
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert not (terminated or truncated)

    envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert terminated or truncated
    assert np.array_equal(np.array([[1.0]], dtype=np.float32), observation)

    # If not manually reset it keeps returning the terminal step
    envs.step(env_ids=[0], actions=np.array([[0.0]], dtype=np.float32))
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert terminated or truncated
    assert np.array_equal(np.array([[1.0]], dtype=np.float32), observation)

    envs.reset(env_ids=[0])
    ids, observation, reward, terminated, truncated, info = envs.gather()
    assert np.array_equal(np.array([[0.0]], dtype=np.float32), observation)

    assert not (terminated or truncated)
    assert "reset" in info
