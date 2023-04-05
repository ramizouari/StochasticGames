import tensorflow_probability as tfp
import tensorflow as tf
from tensorflow.python.framework.indexed_slices import tensor_util
from tensorflow_probability.python.internal import assert_util, parameter_properties


class DiscreteUniform(tfp.distributions.Distribution):
    def __init__(self, low, high, validate_args=False, allow_nan_stats=True, name='DiscreteUniform',dtype=tf.int32):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._low = tf.convert_to_tensor(low, name='low')
            self._high = tf.convert_to_tensor(high, name='high')
            if dtype is None:
                dtype = self._low.dtype
            if dtype != self._low.dtype:
                self._low = tf.cast(self._low, dtype=dtype)
                self._high = tf.cast(self._high, dtype=dtype)
            if dtype not in [tf.int32, tf.int64, tf.int16, tf.int8, tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
                raise TypeError("dtype must be an integer type")
            super(DiscreteUniform, self).__init__(
                dtype=dtype,
                reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                name=name)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(tf.shape(self.low), tf.shape(self.high))

    def _batch_shape(self):
        return tf.broadcast_static_shape(self.low.shape, self.high.shape)

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])


    def _log_prob(self, x):
        return tf.where(
            tf.logical_and(tf.greater_equal(x, self.low), tf.less_equal(x, self.high)),
            tf.zeros_like(x),
            tf.fill(tf.shape(x), float('-inf')))

    def _prob(self, x):
        return tf.where(
            tf.logical_and(tf.greater_equal(x, self.low), tf.less_equal(x, self.high)),
            tf.ones_like(x),
            tf.zeros_like(x))

    def _cdf(self, x):
        return tf.where(
            tf.logical_and(tf.greater_equal(x, self.low), tf.less_equal(x, self.high)),
            tf.ones_like(x),
            tf.zeros_like(x))

    def _entropy(self):
        return tf.log(self.high - self.low + 1)

    def _mean(self):
        return (self.high + self.low) / 2

    def _variance(self):
        return (self.high - self.low + 1) / 12

    def _mode(self):
        return tf.floor((self.high + self.low) / 2)

    def _stddev(self):
        return tf.sqrt(self.variance())

    def _quantile(self, p):
        return tf.floor(p * (self.high - self.low + 1) + self.low)

    def _default_event_space_bijector(self):
        return tfp.bijectors.AffineScalar(shift=self.low)

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        if is_init != tensor_util.is_ref(self.low):
            assertions.append(assert_util.assert_non_negative(
                self.low, message='Argument `low` must be non-negative.'))
        if is_init != tensor_util.is_ref(self.high):
            assertions.append(assert_util.assert_non_negative(
                self.high, message='Argument `high` must be non-negative.'))
        return assertions

    @property
    def low(self):
        """Lower bound of the distribution."""
        return self._low

    @property
    def high(self):
        """Upper bound of the distribution."""
        return self._high

    def _parameter_properties(self, dtype, num_classes=None):
        return dict(
            low=parameter_properties.ParameterProperties(),
            high=parameter_properties.ParameterProperties())

    def _sample_control_dependencies(self, x):
        if not self.validate_args:
            return []
        assertions = []
        assertions.append(assert_util.assert_less_equal(
            x, self.high, message='Sample must be less than or equal to high.'))
        assertions.append(assert_util.assert_greater_equal(
            x, self.low, message='Sample must be greater than or equal to low.'))
        return assertions

    def _log_prob_control_dependencies(self, x):
        if not self.validate_args:
            return []
        assertions = []
        assertions.append(assert_util.assert_less_equal(
            x, self.high, message='Sample must be less than or equal to high.'))
        assertions.append(assert_util.assert_greater_equal(
            x, self.low, message='Sample must be greater than or equal to low.'))
        return assertions

    def _prob_control_dependencies(self, x):
        if not self.validate_args:
            return []
        assertions = []
        assertions.append(assert_util.assert_less_equal(
            x, self.high, message='Sample must be less than or equal to high.'))
        assertions.append(assert_util.assert_greater_equal(
            x, self.low, message='Sample must be greater than or equal to low.'))
        return assertions

    def _cdf_control_dependencies(self, x):
        if not self.validate_args:
            return []
        assertions = []
        assertions.append(assert_util.assert_less_equal(
            x, self.high, message='Sample must be less than or equal to high.'))
        assertions.append(assert_util.assert_greater_equal(
            x, self.low, message='Sample must be greater than or equal to low.'))
        return assertions

    def sample(self, sample_shape=(), seed=None, name='sample'):
        """Sample `sample_shape` values from this distribution.
        Args:
          sample_shape: `int` `Tensor` shape of the number of samples to draw.
            Default value: `[]` (i.e., scalar sample).
          seed: Python integer to seed the random number generator.
          name: Python `str` name prefixed to Ops created by this method.
            Default value: `'sample'`.
        Returns:
          sample: `sample_shape`-shaped `Tensor` of samples from this
            distribution.
        Raises:
          ValueError: if `sample_shape` is not a scalar.
        """
        with self._name_and_control_scope(name):
            sample_shape = tf.convert_to_tensor(
                sample_shape, dtype=tf.int32, name='sample_shape')
            if sample_shape.shape.ndims ==0:
                sample_shape = tf.stack([sample_shape])
            sample_rank = tf.size(sample_shape)
            params_rank=tf.rank(self._batch_shape())
            X=tf.map_fn(lambda x: tf.random.uniform(sample_shape, x[0], x[1], dtype=self.dtype, seed=seed),
                      tf.stack([self.low, self.high], axis=-1))
            return tf.transpose(X, perm=tf.concat([tf.range(params_rank, sample_rank+params_rank), tf.range(params_rank)], axis=0))