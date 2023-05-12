import tf_agents as tfa
import tensorflow as tf

class MPGNetworkExample(tfa.networks.Network):
    def __init__(self,input_tensor_spec,n_actions,state_spec=(),name="MPGNetworkExample"):
        super().__init__(input_tensor_spec, state_spec=state_spec,name=name)
        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(128,activation=tf.nn.relu)
        self.predictions = tf.keras.layers.Dense(n_actions)


    def call(self,observations:tfa.trajectories.Trajectory,step_type=None, network_state=(),training=False):
        G=observations["environment"]
        A=observations["state"]
        G=tf.reshape(G,[1]+G.shape)
        G=self.flatten(G)
        A=tf.cast(tf.reshape(A,[1,1]),dtype=tf.float32)
        Z=self.concat([G,A])
        Z=self.dense(Z)
        return self.predictions(Z),network_state