import tf_agents as tfa


class MPGDenseNetwork(tfa.networks.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, hidden_layers, activation_fn=tfa.activations.relu,
                 name="MPGDenseNetwork"):
        super().__init__(input_tensor_spec, state_spec=(), name=name)
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.output_tensor_spec = output_tensor_spec
        self._layers = []
        for i in range(len(hidden_layers)):
            self._layers.append(tfa.layers.Dense(hidden_layers[i], activation_fn))
        self._layers.append(tfa.layers.Dense(output_tensor_spec.shape.num_elements()))