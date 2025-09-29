import functools
import pytest

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from parallenv.parallenv import (
    ParallEnv,
    AlreadyPendingEnvError,
    ClosedEnvError,
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

    def __init__(self, id: int = 0):
        self.id = id
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._stepped = False

    def step(self, action):
        # First step after reset: terminate
        if not self._stepped:
            self._stepped = True
            return (
                np.array([0.5], dtype=np.float32),
                1.0,
                True,
                False,
                {"env_id": self.id, "step": 1},
            )
        # Subsequent steps keep returning a normal non-terminal transition
        return (
            np.array([0.7], dtype=np.float32),
            0.5,
            False,
            False,
            {"env_id": self.id, "step": 2},
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._stepped = False
        return np.array([0.1], dtype=np.float32), {"env_id": self.id, "reset": True}


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
