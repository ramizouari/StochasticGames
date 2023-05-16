import abc
from typing import TypedDict

import numpy as np
import tf_agents as tfa
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from mpg.games import MeanPayoffGraph
from mpg.games import strategy as agent_strategy

from . import utils


class TransitionInfo(TypedDict):
    """
    A transition info
    """
    observation: types.Array
    reward: types.Array
    discount: types.Array
    step_type: types.Array
    # Whether the action was successful or not
    action_status: types.Bool


class MPGEnvironment(tfa.environments.py_environment.PyEnvironment):
    """
    MPG environment
    """

    def __init__(self, graph: MeanPayoffGraph, starting_vertex, starting_turn, max_turns, max_actions=None,
                 max_reward=None, discount_factor=1.0, bad_action_penalty=None):
        """
        Create a new MPG environment
        :param graph: The graph
        :param starting_vertex: The starting vertex
        :param starting_turn: The starting turn
        :param max_turns: The maximum number of turns
        :param max_actions: The maximum number of actions
        """
        if max_actions is None:
            max_actions = len(graph)
        if max_reward is None:
            # TODO: This is should be a function of the graph
            max_reward = 1
        self.count_vertices = len(graph)
        super(MPGEnvironment, self).__init__()
        self.graph = graph
        self.starting_vertex = starting_vertex
        self.starting_turn = starting_turn
        self.max_turns = max_turns
        self.max_actions = max_actions
        self._action_spec = tfa.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.max_actions - 1, name='action')
        self._observation_spec = tfa.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.count_vertices - 1, name='observation')
        self._episode_ended = False
        self._turn = starting_turn
        self._vertex = starting_vertex
        self._reward = 0
        self.discount_factor = discount_factor
        self._time_step = 0
        self.allow_bad_actions = bad_action_penalty is not None
        self.bad_action_penalty = bad_action_penalty
        self._info = None

    def _step(self, action):
        """
        Take a step in the environment
        :param action: The action to take
        :return: The transition
        """
        if self._episode_ended:
            return self.reset()

        if int(action) not in self.graph.succ[int(self._vertex)]:
            if self.allow_bad_actions:
                self._time_step += 1
                self._reward = self.bad_action_penalty
                self._info = TransitionInfo(observation=self._vertex, reward=self._reward,
                                            discount=self.discount_factor,
                                            action_status=False, step_type=ts.StepType.MID)
                return tfa.trajectories.time_step.transition(self._vertex, self._reward, self.discount_factor)
            else:
                raise ValueError(
                    f"Action {action} should be an adjacent vertex to the current one. current vertex is {self._vertex}")
        if self.max_turns is not None and self._time_step >= self.max_turns:
            self._time_step += 1
            self._episode_ended = True
            self._info = TransitionInfo(observation=self._vertex, reward=self._reward, discount=self.discount_factor,
                                        action_status=True, step_type=ts.StepType.LAST)
            return tfa.trajectories.time_step.termination(self._vertex, self._reward)
        self._time_step += 1
        self._reward = self.graph.edges[int(self._vertex), int(action)]["weight"]
        self._vertex = np.array(action, dtype=np.int32)
        self._info = TransitionInfo(observation=self._vertex, reward=self._reward, discount=self.discount_factor,
                                    action_status=True, step_type=ts.StepType.MID)
        return tfa.trajectories.time_step.transition(self._vertex, self._reward, self.discount_factor)

    def _reset(self):
        """
        Reset the environment
        :return: The initial time step
        """
        self._episode_ended = False
        self._time_step = 0
        self._turn = self.starting_turn
        self._vertex = np.array(self.starting_vertex, dtype=np.int32)
        self._reward = 0
        return tfa.trajectories.time_step.restart(self._vertex)

    def action_spec(self):
        """
        Get the action spec
        :return: The action spec
        """
        return self._action_spec

    def observation_spec(self):
        """
        Get the observation spec
        :return: The observation spec
        """
        return self._observation_spec

    def get_state(self):
        return self._vertex

    def get_info(self) -> types.NestedArray:
        return self._info

    @property
    def game(self):
        return self.graph

    def get_game(self):
        return self.game

    def get_graph(self):
        return self.graph

    @property
    def weights_matrix(self):
        return self.graph.weights_matrix

    @property
    def adjacency_matrix(self):
        return self.graph.adjacency_matrix

    def increment_time(self):
        self._time_step += 1

    def decrement_time(self):
        self._time_step -= 1

PartiallyObservableMPGEnvironment = MPGEnvironment

class FixedStrategyMPGEnvironment(tfa.environments.py_environment.PyEnvironment):
    """
    An MPG environment where the second player has a fixed strategy
    """

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            self.reset()

        current_state = self.env.get_state()
        intermediate_state = self.env.step(action)
        if self.max_turns is not None and self._time_step >= self.max_turns:
            step_type = ts.StepType.LAST
        else:
            step_type = intermediate_state.step_type
        if self.env.get_info()["action_status"]:
            next_action = self.strategy(int(intermediate_state.observation))
            next_state = self.env.step(next_action)
            if not self.env.get_info()["action_status"]:
                raise ValueError(
                    f"Fixed strategy returned an invalid action {next_action} from state {intermediate_state.observation}")
        else:
            next_state = intermediate_state
            self.env.increment_time()
        return tfa.trajectories.time_step.TimeStep(observation=next_state.observation,
                                                   reward=intermediate_state.reward + next_state.reward,
                                                   discount=intermediate_state.discount,
                                                   step_type=step_type)

    def _reset(self) -> ts.TimeStep:
        self._time_step = 0
        return self.env.reset()

    def __init__(self, env: MPGEnvironment, strategy: agent_strategy.Strategy, max_turns: int = None):
        """
        Create a new MPG environment
        :param env: The underlying MPG environment
        :param max_turns: The maximum number of turns
        """
        super(FixedStrategyMPGEnvironment, self).__init__()
        self.env = env
        self.strategy = strategy
        self.max_turns = max_turns if max_turns is not None else env.max_turns
        self.max_actions=env.max_actions
        self._action_spec = tfa.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.env.max_actions - 1, name='action')
        self._observation_spec = tfa.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.env.count_vertices - 1, name='observation')
        self._episode_ended = False
        self._turn = self.env.starting_turn
        self._vertex = self.env.starting_vertex
        self._reward = 0
        self._time_step = 0
        self.count_vertices = self.env.count_vertices
        self.starting_turn=self.env.starting_turn
        self.starting_vertex=self.env.starting_vertex

    def get_state(self):
        return self.env.get_state()

    @property
    def graph(self):
        return self.env.graph

    @property
    def game(self):
        return self.env.graph

    def get_game(self):
        return self.game

    def get_graph(self):
        return self.graph


class FullyObservableMPGEnvironment(tfa.environments.py_environment.PyEnvironment):
    def __init__(self, env: MPGEnvironment, max_turns: int = None):
        """
        Create a fully observable MPG environment from an MPG environment
        :param env: The underlying MPG environment
        :param max_turns: The maximum number of turns
        """
        super(FullyObservableMPGEnvironment, self).__init__()
        self.env = env
        self.max_turns = max_turns if max_turns is not None else env.max_turns
        self.count_vertices = env.count_vertices
        self._action_spec = tfa.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.env.max_actions - 1, name='action')
        self._observation_spec = {"state":tfa.specs.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.count_vertices-1, name='observation'),
            "environment":tfa.specs.ArraySpec(shape=(2,self.count_vertices,self.count_vertices), dtype=np.float32, name='environment')}
        self._episode_ended = False
        self._turn = self.env.starting_turn
        self._vertex = self.env.starting_vertex
        self._reward = 0
        self._time_step = 0
        self.max_actions = env.max_actions

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            self.reset()
            self._episode_ended = False
        next_state = self.env.step(action)
        if next_state.step_type == ts.StepType.LAST:
            self._episode_ended = True
        return tfa.trajectories.time_step.TimeStep(observation=utils.encode_fully_observable_state(self.env, next_state.observation),
                                                   reward=next_state.reward,
                                                   discount=next_state.discount,
                                                   step_type=next_state.step_type)

    def _reset(self) -> ts.TimeStep:
        self._time_step = 0
        initial_state=self.env.reset()
        return tfa.trajectories.time_step.TimeStep(observation=utils.encode_fully_observable_state(self.env, initial_state.observation),
                                                   reward=initial_state.reward,
                                                   discount=initial_state.discount,
                                                   step_type=initial_state.step_type)

class MPGEnvironmentExtractor(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def call(self, env):
        pass

    def __call__(self, env: MPGEnvironment):
        return self.call(env)

    @property
    def env_specs(self) -> tfa.specs.ArraySpec:
        return self.get_env_specs(env=None)

    @abc.abstractmethod
    def get_env_specs(self, env: MPGEnvironment = None) -> tfa.specs.ArraySpec:
        pass


class MPGGraphExtractor(MPGEnvironmentExtractor):
    def __init__(self):
        super().__init__()

    def call(self, env: MPGEnvironment):
        return env.graph


class MPGMatrixExtractor(MPGEnvironmentExtractor):
    def __init__(self, matrix="both", graph_size=None):
        super().__init__()
        self.matrix = matrix
        self.graph_size = graph_size

    def call(self, env: MPGEnvironment):
        return utils.environment_representation(env, representation=self.matrix)

    def get_env_shape(self, env=None):
        if env is None:
            if self.graph_size is None:
                vertex_count = None
            else:
                vertex_count = self.graph_size
        else:
            vertex_count = len(env.get_game())
        shape = (vertex_count, vertex_count)
        if self.matrix == "both":
            shape = (2, vertex_count, vertex_count)
        return shape

    def get_env_specs(self, env=None) -> tfa.specs.ArraySpec:
        return tfa.specs.ArraySpec(shape=self.get_env_shape(env), dtype=np.float32)


class MPGTrajectoryConverter:
    def __init__(self, env: MPGEnvironment, extractor: MPGEnvironmentExtractor, add_batch_dim=False):
        self.env = env
        self.extractor = extractor
        self.add_batch_dim = add_batch_dim

    def __call__(self, trajectory: tfa.trajectories.Trajectory):
        return self.convert(trajectory)

    def convert(self, trajectory: tfa.trajectories.Trajectory):
        trajectory = tfa.trajectories.Trajectory(
            step_type=trajectory.step_type,
            observation={"environment": self.extractor(self.env), "state": trajectory.observation},
            action=trajectory.action,
            policy_info=trajectory.policy_info,
            next_step_type=trajectory.next_step_type,
            reward=trajectory.reward,
            discount=trajectory.discount
        )
        #        return trajectory if not self.add_batch_dim else tfa.utils.nest_utils.batch_nested_tensors(trajectory)
        return trajectory.observation if not self.add_batch_dim else tfa.utils.nest_utils.batch_nested_tensors(
            trajectory.observation)

    @property
    def data_spec(self):
        return {"environment": self.extractor.env_specs, "state": self.env.observation_spec()}
