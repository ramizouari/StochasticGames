import abc
from typing import Callable, List

import tf_agents as tfa
from . import environment


class MPGDriver(tfa.drivers.py_driver.PyDriver):

    def mapper(self, observer: Callable[[tfa.trajectories.Trajectory], tfa.trajectories.Trajectory]) -> Callable[
        [tfa.trajectories.Trajectory], tfa.trajectories.Trajectory]:
        return lambda trajectory: observer(self.converter.convert(trajectory))

    def mappers(self, observers: List[Callable[[tfa.trajectories.Trajectory], tfa.trajectories.Trajectory]]) -> List[
        Callable[[tfa.trajectories.Trajectory], tfa.trajectories.Trajectory]]:
        return [self.mapper(observer) for observer in observers]

    def __init__(self, env:environment.MPGEnvironment, policy, total_observers, partial_observers, transition_observers=None, info_observers=None,
                num_episodes=1, max_steps=1,
                extractor: environment.MPGEnvironmentExtractor = None):
        if extractor is None:
            extractor = environment.MPGMatrixExtractor(matrix="both")
        self.converter = environment.MPGTrajectoryConverter(env, extractor,add_batch_dim=True)
        observers = self.mappers(total_observers) + partial_observers
        super().__init__(env, policy, observers, transition_observers, info_observers, num_episodes, max_steps)
