from typing import Optional

import numpy as np
import tf_agents.policies.py_policy
from tf_agents.trajectories import policy_step
from tf_agents.typing import types as tfa_types

from ..games import strategy
import tf_agents as tfa
import tf_agents.trajectories.time_step as ts


class StrategyWrapper(tfa.policies.py_policy.PyPolicy):
    def __init__(self, strategy: strategy.AbstractStrategy):
        self.strategy = strategy
        time_spec = None
        count_vertices = len(strategy.get_game())
        action_spec = tfa.specs.ArraySpec(shape=(), dtype=np.int32, name="action", maximum=count_vertices - 1,
                                          minimum=0)
        super().__init__(time_spec, action_spec, policy_state_spec=(), info_spec=())

    def _action(self,
                time_step: ts.TimeStep,
                policy_state: tfa_types.NestedArray,
                seed: Optional[tfa_types.Seed] = None) -> policy_step.PolicyStep:
        return policy_step.PolicyStep(self.strategy(time_step.observation), policy_state)


class ApproximateStrategy(tf_agents.policies.py_policy.PyPolicy):
    def __init__(self, policy:tf_agents.policies.py_policy.PyPolicy, epsilon: float,seed = None):
        self.policy = policy
        self.epsilon = epsilon
        super().__init__(policy.time_step_spec, policy.action_spec, policy_state_spec=policy.policy_state_spec,
                         info_spec=policy.info_spec)
        if seed is None:
            seed=np.random.randint(0, 2**32-1)
        self.rng=np.random.default_rng(seed)

    def _action(self,
                time_step: ts.TimeStep,
                policy_state: tfa_types.NestedArray,
                seed: Optional[tfa_types.Seed] = None) -> policy_step.PolicyStep:
        if self.rng.random() < self.epsilon:
            return policy_step.PolicyStep(np.random.randint(0, len(self.policy.get_game())), policy_state)
        else:
            return self.policy.action(time_step, policy_state, seed)


