from typing import Tuple, List, Callable

from . import environment
import tf_agents as tfa


class Arena:
    def __init__(self,env:environment.MPGEnvironment,policies:Tuple[tfa.policies.py_policy.PyPolicy,tfa.policies.py_policy.PyPolicy],
                 num_episodes=1, max_steps=1):
        self.env = env
        self.policies = policies
        self.num_episodes = num_episodes
        self.max_steps = max_steps

# TODO: Implement an arena class for comparing agents
class PyArena(Arena):

    def __init__(self,env:environment.MPGEnvironment,policies:Tuple[tfa.policies.py_policy.PyPolicy,tfa.policies.py_policy.PyPolicy],
                 num_episodes=1, max_steps=1,*,
                 p1_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 p2_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 p1_transition_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 p2_transition_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 p1_info_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 p2_info_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 transition_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None,
                 info_observers:List[Callable[[tfa.trajectories.Trajectory],tfa.trajectories.Trajectory]] = None):
        super().__init__(env,policies,num_episodes,max_steps)
        self.p1_observers=p1_observers + observers
        self.p2_observers=p2_observers + observers
        self.p1_transition_observers=p1_transition_observers + transition_observers
        self.p2_transition_observers=p2_transition_observers + transition_observers
        self.p1_info_observers=p1_info_observers + info_observers
        self.p2_info_observers=p2_info_observers + info_observers


    def run(self):
        for ep in range(self.num_episodes):
            time_step = self.env.reset()
            for step in range(self.max_steps):
                for observer in self.p1_observers:
                    observer(self.env.trajectory)
                for observer in self.p1_info_observers:
                    observer(self.env.trajectory)
                action1 = self.policies[0].action(time_step)
                for observer in self.p1_transition_observers:
                    observer(self.env.trajectory)

                time_step = self.env.step(action1)

                for observer in self.p2_observers:
                    observer(self.env.trajectory)
                for observer in self.p2_info_observers:
                    observer(self.env.trajectory)
                action2 = self.policies[1].action(time_step)
                for observer in self.p2_transition_observers:
                    observer(self.env.trajectory)

                for observer in self.p2_transition_observers:
                    observer(self.env.trajectory)
                time_step = self.env.step(action2)


class MCTSArena(Arena):
    def __init__(self, env: environment.MPGEnvironment,
                 policies: Tuple[tfa.policies.py_policy.PyPolicy, tfa.policies.py_policy.PyPolicy],
                 num_episodes=1, max_steps=1):
        super().__init__(env, policies, num_episodes, max_steps)

    def run(self):
        raise NotImplementedError
