from typing import Optional

import numpy as np
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
