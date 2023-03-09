from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates import jax as tfp


class ParamsMeanField(NamedTuple):
    mus: Any
    stddev: Any

    def log_prob(self, params):
        dist, self_flat_params, _ = self._to_dist()
        flat_params, _ = ravel_pytree(params)
        if len(flat_params) != len(self_flat_params):
            quotient, remainder = divmod(len(flat_params), len(self_flat_params))
            assert (
                remainder == 0
            ), "Given parameters are not given in the form of batches of parameters."
            flat_params = flat_params.reshape((quotient, len(self_flat_params)))
        return dist.log_prob(flat_params)

    def sample(self, seed, n_samples):
        dist, _, pytree_def = self._to_dist()
        samples = dist.sample(seed=seed, sample_shape=(n_samples,))
        pytree_def = jax.vmap(pytree_def)
        return pytree_def(samples)

    def _to_dist(
        self,
    ):
        self_flat_mus, pytree_def = ravel_pytree(self.mus)
        self_flat_stddevs, _ = ravel_pytree(self.stddev)
        dist = tfp.distributions.MultivariateNormalDiag(
            self_flat_mus, jnp.ones_like(self_flat_mus) * self_flat_stddevs
        )
        return dist, self_flat_mus, pytree_def
