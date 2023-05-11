import tf_agents as tfa
import types

from . import utils, environment


class FullyObservableMPGAgentWrapper:
    def __new__(cls, agent: tfa.agents.TFAgent, env: environment.MPGEnvironment, **kwargs):
        agent.preprocess_sequence = types.MethodType(cls.preprocess_sequence, agent)
        agent.current_environment = env
        return agent

    @staticmethod
    def preprocess_sequence(self, experience: tfa.trajectories.Trajectory):
        return tfa.trajectories.Trajectory(observation=utils.encode_fully_observable_state(self.current_environment,state=experience.observation),
                                           step_type=experience.step_type, action=experience.action,
                                           policy_info=experience.policy_info,
                                           reward=experience.reward,
                                           next_step_type=experience.next_step_type,
                                           discount=experience.discount)
