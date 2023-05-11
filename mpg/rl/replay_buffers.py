import numpy as np
import tf_agents as tfa


class MPGMatrixBuffer(tfa.replay_buffers.py_uniform_replay_buffer.PyUniformReplayBuffer):
    def __init__(self, env_specs, capacity, ):
        data_spec = {"state": tfa.specs.ArraySpec(shape=(), dtype=np.int32),
                     "environment": env_specs}

        super().__init__(data_spec, capacity)

