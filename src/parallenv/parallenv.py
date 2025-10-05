from collections.abc import Callable, Sequence
from enum import Enum
from math import ceil
import multiprocessing as mp
from multiprocessing import connection
import queue
import threading
from typing import Any, Generic, TypeVar
from warnings import warn

import gymnasium as gym
from gymnasium.vector.utils import batch_space, concatenate, create_empty_array
import numpy as np


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType", bound=np.ndarray)

T = TypeVar("T")

Seq = list[Any] | np.ndarray
Ids = list[int] | np.ndarray


class AutoresetMode(Enum):
    """Enum representing the different autoreset modes, next step, same step and disabled."""

    NEXT_STEP = "NextStep"
    SAME_STEP = "SameStep"
    DISABLED = "Disabled"


class EnvState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"


class ClosedEnvError(Exception): ...


class AlreadyPendingEnvError(Exception): ...


class ClosingSentinel: ...


def split_n(x, n: int):
    size = ceil(len(x) // n)
    return [x[i * size : (i + 1) * size] for i in range(n)]


class ParallEnv(Generic[ObsType, ActType]):
    """A parallel execution wrapper for Gymnasium's Envs inspired by its vector API.

    This class does not implement the :class:`~gymnasium.vector.VectorEnv` API. It is inspired by
    them to promote compatibility of spaces, metadata and info aggregation, but its purpose is different: it wraps
    multiple sub-environments and makes it easier to parallelize their execution together with a policy by keeping
    :meth:`reset` and :meth:`step` non-blocking. Work is enqueued, and results are asynchronously collected as fixed-size
    batches via :meth:`gather`.

    Despite not following the VecEnv API, returned batches follow Gymnasium's vector conventions: observations
    are concatenated according to ``single_observation_space``, rewards/terminations/truncations are 1-D numpy arrays,
    and ``infos`` is a dictionary aggregated with per-key boolean masks (e.g. ``_key``). Depending on the selected
    :class:`~gymnasium.vector.AutoresetMode`, ``infos`` may include ``final_obs`` and ``final_info`` when episodes end.

    Key differences from standard vector envs:
    - :meth:`reset` and :meth:`step` do not return observations immediately and do not block, and additional `env_ids`
      argument is used to control which sub-environments receive the commands. :meth:`gather` is used to
      retrieve the next available batch.
    - Experience batches mixes both `step` and `reset` generated data, so a freshly reset environment corresponding
      data will have values while in the Gymnaisum Env API reset only produces `observation` and `info` (see :meth:`gather`
      for more information)
    - Batch size is controlled by `batch_size`, not by `num_envs`, by having a smaller `batch_size` than `num_envs`
      it is possible to run environments at the same time than policies.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], gym.Env]],
        batch_size: int,
        num_workers: int,
        daemon: bool = True,
        autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
    ):
        """Create a parallel environment that yields fixed-size batches.

        Args:
            env_fns: Callables that construct the sub-environments. Environments are sharded across worker processes;
                a single worker may manage several environments.
            batch_size: Number of experiences to aggregate per output batch returned by :meth:`gather`.
            num_workers: Number of worker processes. Sub-environments are split across workers.
            daemon: Whether worker processes are started as daemons.
            autoreset_mode: Autoreset strategy used by sub-environments. See
                :class:`~gymnasium.vector.AutoresetMode` and Farama docs for details.

        Notes:
            - Operations are non-blocking: :meth:`reset` and :meth:`step` enqueue work; call :meth:`gather` to receive
              the next available batch of size ``batch_size``.
            - ``action_space`` and ``observation_space`` are batched for ``batch_size`` items (not ``num_envs``).
            - ``metadata["autoreset_mode"]`` is set to the chosen mode for downstream wrappers/tools.
        """
        self.env_fns = env_fns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.daemon = daemon
        self.autoreset_mode = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )

        self.closed = False
        self._has_reset = False
        self.num_envs = len(env_fns)
        self.envs_ids = list(range(self.num_envs))
        self.workers_envs_ids = split_n(self.envs_ids, self.num_workers)

        self._envs_states = [EnvState.DEFAULT for _ in range(self.num_envs)]

        dummy_env = env_fns[0]()

        self.metadata = dummy_env.metadata
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.render_mode = dummy_env.render_mode

        self.single_action_space = dummy_env.action_space
        self.action_space = batch_space(self.single_action_space, self.batch_size)

        # This is like the "same" observation_mode in gym.vector.VectorEnv classes
        self.single_observation_space = dummy_env.observation_space
        self.observation_space = batch_space(
            self.single_observation_space, self.batch_size
        )

        dummy_env.close()
        del dummy_env

        workers_envs_fns = [
            [env_fns[i] for i in worker_envs_ids]
            for worker_envs_ids in self.workers_envs_ids
        ]

        self.experience_queue = mp.SimpleQueue()
        self.batches_queue = queue.Queue()

        self.command_queues: list[mp.SimpleQueue] = []
        self.processes: list[mp.Process] = []

        for worker_idx in range(self.num_workers):
            command_queue = mp.SimpleQueue()
            process = mp.Process(
                target=_parallel_worker,
                name=f"Worker<{type(self).__name__}>-{worker_idx}",
                args=(
                    workers_envs_fns[worker_idx],
                    command_queue,
                    self.experience_queue,
                    self.workers_envs_ids[worker_idx],
                    self.autoreset_mode,
                ),
            )
            self.command_queues.append(command_queue)
            self.processes.append(process)

            process.daemon = daemon
            process.start()

        self.consumer_worker = threading.Thread(target=self._consumer_loop, daemon=True)
        self.consumer_worker.start()

    def _add_info(
        self, vector_infos: dict[str, Any], env_info: dict[str, Any], env_num: int
    ) -> dict[str, Any]:
        """Add env info to the info dictionary of the vectorized environment.

        Given the `info` of a single environment add it to the `infos` dictionary
        which represents all the infos of the vectorized environment.
        Every `key` of `info` is paired with a boolean mask `_key` representing
        whether or not the i-indexed environment has this `info`.

        Args:
            vector_infos (dict): the infos of the vectorized environment
            env_info (dict): the info coming from the single environment
            env_num (int): the index of the single environment

        Returns:
            infos (dict): the (updated) infos of the vectorized environment
        """
        # This is copied-pasted from gym.vector.VectorEnv
        for key, value in env_info.items():
            # It is easier for users to access their `final_obs` in the unbatched array of `obs` objects
            if key == "final_obs":
                if "final_obs" in vector_infos:
                    array = vector_infos["final_obs"]
                else:
                    array = np.full(self.num_envs, fill_value=None, dtype=object)
                array[env_num] = value
            # If value is a dictionary, then we apply the `_add_info` recursively.
            elif isinstance(value, dict):
                array = self._add_info(vector_infos.get(key, {}), value, env_num)
            # Otherwise, we are a base case to group the data
            else:
                # If the key doesn't exist in the vector infos, then we can create an array of that batch type
                if key not in vector_infos:
                    if type(value) in [int, float, bool] or issubclass(
                        type(value), np.number
                    ):
                        array = np.zeros(self.num_envs, dtype=type(value))
                    elif isinstance(value, np.ndarray):
                        # We assume that all instances of the np.array info are of the same shape
                        array = np.zeros(
                            (self.num_envs, *value.shape), dtype=value.dtype
                        )
                    else:
                        # For unknown objects, we use a Numpy object array
                        array = np.full(self.num_envs, fill_value=None, dtype=object)
                # Otherwise, just use the array that already exists
                else:
                    array = vector_infos[key]

                # Assign the data in the `env_num` position
                #   We only want to run this for the base-case data (not recursive data forcing the ugly function structure)
                array[env_num] = value

            # Get the array mask and if it doesn't already exist then create a zero bool array
            array_mask = vector_infos.get(
                f"_{key}", np.zeros(self.num_envs, dtype=np.bool_)
            )
            array_mask[env_num] = True

            # Update the vector info with the updated data and mask information
            vector_infos[key], vector_infos[f"_{key}"] = array, array_mask
        return vector_infos

    def _consumer_loop(self) -> None:
        """Collects and batches the experiences generated by the environments.

        This is meant to be executed concurrently by a secondary thread.
        """
        id_buffer = []
        observation_buffer, rewards, terminations, truncations, infos = (
            [],
            [],
            [],
            [],
            {},
        )
        ob_idx = 0
        close_sentinels = 0
        while True:
            received = self.experience_queue.get()
            if isinstance(received, ClosingSentinel):
                close_sentinels += 1
                if close_sentinels == self.num_workers:
                    break
                else:
                    continue
            id_, (observation, reward, terminated, truncated, info) = received
            id_buffer.append(id_)
            observation_buffer.append(observation)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            # TODO: be careful with ob_idx, study how _add_info works
            infos = self._add_info(infos, info, ob_idx)
            ob_idx += 1
            # TODO: consider controlling the batch extraction
            if len(id_buffer) >= self.batch_size:
                observations = create_empty_array(
                    self.single_observation_space, n=len(id_buffer), fn=np.zeros
                )
                observations = concatenate(
                    self.single_observation_space,
                    observation_buffer,
                    observations,
                )
                self.batches_queue.put(
                    (
                        np.array(id_buffer, dtype=np.int32),
                        observations,
                        np.array(rewards, dtype=np.float64),
                        np.array(terminations, dtype=np.bool_),
                        np.array(truncations, dtype=np.bool_),
                        infos,
                    )
                )
                id_buffer = []
                observation_buffer, rewards, terminations, truncations, infos = (
                    [],
                    [],
                    [],
                    [],
                    {},
                )
                ob_idx = 0

    def _filter_seq(self, seq: Seq, mask: list[bool] | np.ndarray) -> Seq:
        """A helper method to filter a sequence with a mask."""
        if isinstance(seq, np.ndarray):
            return seq[mask]
        filt_seq = [value for value, flag in zip(seq, mask) if flag]
        return filt_seq

    def _split_by_workers(
        self, env_ids: Ids, seqs: list[Seq] | None = None
    ) -> tuple[list[int], list[Ids], list[list[Seq]]]:
        """Assigns environment ids to their workers and splits sequences accordingly.

        Assigns an environment ids sequence to their correspondant worker, it can be
        used also to split any arbitrary sequence associated with the ids sequence.

        Args:
            env_ids: A list (or array) of ids to be assigned to workers and split.
            seqs: A list of sequences with the same length as ids to be split.

        Returns:
            A tuple containing:
                A list with the workers ids (those ids are the indexes of workers)
                  that conains all of the passed `env_ids`. There may be a variable
                  number of workers ids, for example, if there are 4 workers in
                  the ParallEnv but passed `env_ids` come from 2 workers, those
                  2 workers ids will be returned.
                The split ids as a list of lists, for each worker id there is a
                  list with the ids of environments in the the correspondant worker.
                A list with the sequences provided in `seqs` split by worker, where each
                  sequence is split exactly as `env_ids`.

        Raises:
            ValueError: If any of the sequences in `seqs` has a different length from
              `env_ids`.
        """
        per_worker_masks = [
            [id_ in worker_env_ids for id_ in env_ids]
            for worker_env_ids in self.workers_envs_ids
        ]
        per_worker_ids = [
            self._filter_seq(env_ids, worker_mask) for worker_mask in per_worker_masks
        ]
        worker_indexes = [
            i for i, worker_env_ids in enumerate(per_worker_ids) if len(worker_env_ids)
        ]
        per_worker_ids = [
            worker_env_ids for worker_env_ids in per_worker_ids if len(worker_env_ids)
        ]
        per_worker_masks = [
            mask for i, mask in enumerate(per_worker_masks) if i in worker_indexes
        ]

        if seqs is None:
            return worker_indexes, per_worker_ids, []

        if any(len(seq) != len(env_ids) for seq in seqs):
            raise ValueError("All sequences in `seqs` must be the same length as `ids`")
        per_worker_seqs = []
        for seq in seqs:
            per_worker_seqs.append(
                [self._filter_seq(seq, worker_mask) for worker_mask in per_worker_masks]
            )
        return worker_indexes, per_worker_ids, per_worker_seqs

    def _check_raise_env_ids(self, env_ids: Sequence[int] | np.ndarray) -> None:
        """Checks environment ids validity.

        Args:
            env_ids: The environment ids to be tested.
        Raises:
            ValueError: If any of the `env_ids` is not in the range of existing env ids.
            ValueError: If there are repeated ids.
        """
        if not all(0 <= id_ < self.num_envs for id_ in env_ids):
            raise ValueError(
                f"`ids` must correspond to existing environment ids in range [0, {self.num_envs - 1}]."
            )
        if not len(env_ids) == len(set(env_ids)):
            raise ValueError("`ids` must not contain repeated values.")

    def _check_envs_states_default(self, env_ids: Sequence[int] | np.ndarray) -> bool:
        """Check if given environments are in default state.

        Args:
            env_ids: The environment ids to be tested.

        Returns:
            True if all environments are in default state.
        """
        return all(self._envs_states[id_] == EnvState.DEFAULT for id_ in env_ids)

    def reset(
        self,
        env_ids: Ids | None = None,
        *,
        seed: int | list[int | None] | np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Resets selected sub-environments.

        Specific sub-environments can be reset, while this is useful, especially
        if autoreset is disable, the first time reset is call it ha to
        be applied to all the sub-environments, if not further `gather` calls could
        block forever, in the case that less environments that the `batch_size` are
        reset. Hence, although `env_ids` (or `reset_mask` in options) can be passed
        to the method during the first reset it will be ignored and all sub-environments
        will be reset.

        Args:
            env_ids: Id of environments to be reset. If not passed (or explicitly passed None)
              then all sub-environments are reset.
            seed: The environment reset seed, either
              * ``None`` - random seeds for all passed environment ids.
              * ``int`` - ``[seed, seed+1, ..., seed+n]``
              * List of ints - ``[1, 2, 3, ..., n]``
              If it is a list, it must be the same length as `env_ids` if it is passed or
              the same length as the number of environments if `env_ids` is not passed.
            options: Option information for sub-environments. The same options will be passed
              to all reset sub-environments, except the optional `reset_mask` field, this field's
              value must be a list with the same length as `env_ids` and it may be used
              to control which sub-environments are reset. This field is not necessary to
              control sub-environemnts reset and the `env_ids` argument is preferred.

        Raises:
            ClosedEnvError: If the environment has already been closed.
            AlreadyPendingEnvError: If any of `env_ids` sub environments is not in default state.
            ValueError: If any id is outside the valid range or if there are
              duplicated ids.
            ValueError: If seeds are passed as a list and its length is different from `env_ids`'s length.

        """
        if self.closed:
            raise ClosedEnvError("Trying to operate on ParallEnv after close().")

        if not self._has_reset:
            if env_ids is None:
                warn(
                    "env_ids has been passed during the first reset, it will be ignored and all sub-environments will be reset instead."
                )
                env_ids = list(range(self.num_envs))
            if options is not None and "reset_mask" in options:
                warn(
                    "reset_mask has been passed during the first reset, it will be ignored and all sub-environments will be reset instead."
                )
                del options["reset_mask"]
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        else:
            self._check_raise_env_ids(env_ids)
        if not self._check_envs_states_default(env_ids):
            raise AlreadyPendingEnvError(
                "There are environments waiting for a pending call"
            )
        for id_ in env_ids:
            self._envs_states[id_] = EnvState.WAITING_RESET
        len_ids = len(env_ids)
        if seed is None:
            seed = [None for _ in range(len_ids)]
        elif isinstance(seed, int):
            seed = [seed + i for i in range(len_ids)]
        else:
            seed = seed
        if len(seed) != len_ids:
            raise ValueError(
                f"If seeds are passed as a list the length must match the length of passed ids (in this case {len_ids}) but got length={len(seed)}."
            )

        # Options is the same for all the environments, as it is in gym.vector.VectorEnv,
        #  except for reset_mask, this option is remove from the options dict here.
        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            assert isinstance(reset_mask, np.ndarray), (
                f"`options['reset_mask': mask]` must be a numpy array, got {type(reset_mask)}"
            )
            env_ids = [id_ for id_, flag in zip(env_ids, reset_mask) if flag]
            seed = [s for s, flag in zip(seed, reset_mask) if flag]
        worker_indexes, per_worker_ids, (per_worker_seeds,) = self._split_by_workers(
            env_ids, [seed]
        )
        for worker_idx, worker_env_ids, worker_seed in zip(
            worker_indexes, per_worker_ids, per_worker_seeds
        ):
            self.command_queues[worker_idx].put(
                ("reset", (worker_env_ids, {"seeds": worker_seed, "options": options}))
            )
        if not self._has_reset:
            self._has_reset = True

    def step(self, env_ids: Ids, actions: ActType) -> None:
        """Queue a step for the given sub-environments.

        This method schedules a non-blocking step call for the provided
        sub-environment `ids` with their corresponding `actions`. Use `gather()`
        to retrieve the next available batched experience tuple.

        Notes:
            Each environment can be involved in only one pending operation at a
              time. Calling `step()` on an environment that hasn't been gathered yet
              will raise an error.
            The returned batch from `gather()` may include experiences generated
              by both `step()` and `reset()` commands, depending on what has been
              scheduled and on the `autoreset_mode`.
            Experiences are emitted in batches of size `batch_size` (see
              constructor). `gather()` will block until a full batch is available or
              a timeout is reached.

        Args:
            ids: Sequence of environment ids to step. All ids must be valid and
                unique.
            actions: Actions aligned positionally with `ids`. It can be any
                sequence or numpy array whose length matches `ids`.

        Raises:
            ClosedEnvError: If the environment has already been closed.
            ValueError: If any id is outside the valid range or if there are
              duplicated ids.
            AlreadyPendingEnvError: If any referenced environment is not in the
              default state (i.e., it has a pending `reset()` or `step()` that
              has not been gathered yet).
        """
        if self.closed:
            raise ClosedEnvError("Trying to operate on ParallEnv after close().")
        self._check_raise_env_ids(env_ids)
        if not self._check_envs_states_default(env_ids):
            raise AlreadyPendingEnvError(
                "There are environments waiting for a pending call"
            )
        for id_ in env_ids:
            self._envs_states[id_] = EnvState.WAITING_STEP
        worker_indexes, per_worker_ids, (per_worker_actions,) = self._split_by_workers(
            env_ids, [actions]
        )
        for worker_idx, worker_env_ids, worker_actions in zip(
            worker_indexes, per_worker_ids, per_worker_actions
        ):
            self.command_queues[worker_idx].put(
                ("step", (worker_env_ids, worker_actions))
            )

    def gather(
        self, timeout: int | None = None
    ) -> tuple[np.ndarray, ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Retrieve the next available batch of experiences.

        Blocks until a batch of size `batch_size` is available in the internal
        queue or until `timeout` is reached. The returned batch aggregates
        results produced by prior `step()` and/or `reset()` requests.

        Args:
            timeout: Optional timeout in seconds to wait for a batch. If `None`,
                waits indefinitely.

        Returns:
            Batch of (env_ids, observations, rewards, terminations, truncations, infos)
              When the batch contains items coming from `reset()`, the
              corresponding entries in `rewards` will be `np.nan` while `terminations`, and
              `truncations` will be False. `infos` will include the reset infos for those entries.
              `infos` is built as in Gymnasium `VectorEnv`classes. Depending on `autoreset_mode`,
              `infos` may also include `final_obs` and `final_info` keys.

        Raises:
            ClosedEnvError: If the environment has already been closed.
            queue.Empty: If `timeout` (is not None and) elapses before a batch becomes available.
        """
        if self.closed:
            raise ClosedEnvError("Trying to operate on ParallEnv after close().")
        batch = self.batches_queue.get(timeout=timeout)
        ids = batch[0]
        for id_ in ids:
            self._envs_states[id_] = EnvState.DEFAULT
        return batch

    def close(self) -> None:
        """Gracefully terminates worker processes and mark the environment closed.

        Sends a close command to all worker processes and waits for their
        termination.


        Notes:
        - Further calls to
          `step()`, `reset()`, or `gather()` after closing are not supported.
        """
        if self.closed:
            return None
        for command_queue in self.command_queues:
            command_queue.put(("close", None))
        connection.wait([p.sentinel for p in self.processes])
        self.consumer_worker.join()
        self.closed = True

    def empty(self):
        return self.batches_queue.empty()


def _step_auto_disable(env: gym.Env, action, autoreset: bool):
    return env.step(action), autoreset


def _step_auto_next_step(env: gym.Env, action, autoreset: bool):
    if autoreset:
        observation, info = env.reset()
        reward, terminated, truncated = 0, False, False
    else:
        observation, reward, terminated, truncated, info = env.step(action)
    autoreset = terminated or truncated
    return (observation, reward, terminated, truncated, info), autoreset


def _step_auto_same_step(env: gym.Env, action, autoreset: bool):
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        reset_observation, reset_info = env.reset()

        info = {
            "final_info": info,
            "final_obs": observation,
            **reset_info,
        }
        observation = reset_observation
    return (observation, reward, terminated, truncated, info), autoreset


step_fn_dict = {
    AutoresetMode.DISABLED: _step_auto_disable,
    AutoresetMode.NEXT_STEP: _step_auto_next_step,
    AutoresetMode.SAME_STEP: _step_auto_same_step,
}


def _parallel_worker(
    env_fns: Sequence[Callable[[], gym.Env]],
    commands_queue: mp.SimpleQueue,
    experience_queue: mp.SimpleQueue,
    envs_ids: list[int],
    autoreset_mode: AutoresetMode,
):
    try:
        step_fn = step_fn_dict[autoreset_mode]
    except KeyError:
        raise ValueError(f"Unexpected autoreset_mode: {autoreset_mode}")

    env_id_to_index = {id_: i for i, id_ in enumerate(envs_ids)}
    envs = [env_fn() for env_fn in env_fns]

    # Autoreset state for NEXT_STEP autoreset mode
    autoreset = [False for _ in envs]

    while True:
        command, data = commands_queue.get()
        if command == "step":
            ids, actions = data
            for i, id_ in enumerate(ids):
                env_index = env_id_to_index[id_]
                env = envs[env_index]
                ((observation, reward, terminated, truncated, info), env_autoreset) = (
                    step_fn(env, actions[i], autoreset[env_index])
                )
                autoreset[env_index] = env_autoreset
                experience_queue.put(
                    (id_, (observation, reward, terminated, truncated, info))
                )
        elif command == "reset":
            ids, all_kwargs = data
            seeds = all_kwargs["seeds"]
            options = all_kwargs["options"]
            for i, id_ in enumerate(ids):
                env_index = env_id_to_index[id_]
                env = envs[env_index]
                observation, info = env.reset(seed=seeds[i], options=options)
                experience_queue.put((id_, (observation, None, None, None, info)))
        elif command == "close":
            break
    for env in envs:
        env.close()
    experience_queue.put(ClosingSentinel())
