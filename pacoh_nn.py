import copy
import functools
from typing import Any, Callable, Iterator, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from jax._src.flatten_util import ravel_pytree
from tensorflow_probability.substrates import jax as tfp

import models


def meta_train(
    data: Iterator[Tuple[npt.ArrayLike, npt.ArrayLike]],
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
    hyper_prior: models.ParamsMeanField,
    hyper_posterior: models.ParamsMeanField,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    iterations: int,
    n_prior_samples: int,
) -> models.ParamsMeanField:
    """Approximate inference of a hyper-posterior, given a hyper-prior and prior.

    Args:
        data (Iterator[Tuple[npt.ArrayLike, npt.ArrayLike]]): The dataset to be learned.
        prediction_fn (Callable[[chex.ArrayTree, chex.Array], chex.Array]): Parameterizd
        function approximator.
        hyper_prior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        hyper_posterior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        optimizer (optax.GradientTransformation): Optimizer.
        opt_state (optax.OptState): Optimizer state.
        iterations (int): Number of update iterations to be performed
        n_prior_samples (int): Number of prior samples to draw for each task.

    Returns:
        models.ParamsMeanField: Trained hyper-posterior.
    """
    hyper_posterior = copy.deepcopy(hyper_posterior)
    keys = hk.PRNGSequence(42)
    for i in range(iterations):
        meta_batch_x, meta_batch_y = next(data)
        hyper_posterior, opt_state, log_probs = train_step(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            hyper_prior,
            hyper_posterior,
            next(keys),
            n_prior_samples,
            optimizer,
            opt_state,
        )
        if i % 100 == 0:
            print(f"Iteration {i} log probs: {log_probs}")
    return hyper_posterior


@functools.partial(jax.jit, static_argnums=(2, 6, 7))
def train_step(
    meta_batch_x,
    meta_batch_y,
    prediction_fn,
    hyper_prior,
    hyper_posterior,
    key,
    n_prior_samples,
    optimizer,
    opt_state,
):
    """Approximate inference of a hyper-posterior, given a hyper-prior and prior.

    Args:
        meta_batch_x: Meta-batch of input data.
        meta_batch_y: Meta-batch of output data.
        prediction_fn: Parameterized function approximator.
        hyper_prior: Prior distribution over distributions of parameterized functions.
        hyper_posterior: Infered posterior distribution over distributions
            parameterized functions.
        key: PRNG key for stochasticity.
        n_prior_samples: Number of prior samples to draw for each task.
        optimizer: Optimizer.
        opt_state: Initial optimizer state.

    Returns:
        Trained hyper-posterior and optimizer state.
    """
    grad_fn = jax.value_and_grad(
        lambda hyper_posterior: particle_loss(
            meta_batch_x,
            meta_batch_y,
            prediction_fn,
            hyper_posterior,
            hyper_prior,
            key,
            n_prior_samples,
        )
    )
    # vmap to compute the grads for each particle in the ensemble with respect
    # to its prediction's log probability.
    log_probs, log_prob_grads = jax.vmap(grad_fn)(hyper_posterior)
    # Compute the particles' kernel matrix and its per-particle gradients.
    num_particles = jax.tree_util.tree_flatten(log_prob_grads)[0][0].shape[0]
    particles_matrix, reconstruct_tree = _to_matrix(hyper_posterior, num_particles)
    kxx, kernel_vjp = jax.vjp(
        lambda x: rbf_kernel(x, particles_matrix), particles_matrix
    )
    # Summing along the 'particles axis' to compute the per-particle gradients, see
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    kernel_grads = kernel_vjp(jnp.ones(kxx.shape))[0]
    stein_grads = (
        jnp.matmul(kxx, _to_matrix(log_prob_grads, num_particles)[0]) + kernel_grads
    ) / num_particles
    stein_grads = reconstruct_tree(stein_grads.ravel())
    updates, new_opt_state = optimizer.update(stein_grads, opt_state)
    new_params = optax.apply_updates(hyper_posterior, updates)
    return (models.ParamsMeanField(*new_params), new_opt_state, log_probs.mean())


def _to_matrix(
    params: chex.ArrayTree, num_particles: int
) -> Tuple[chex.Array, Callable[[chex.Array], hk.Params]]:
    flattened_params, reconstruct_tree = ravel_pytree(params)
    matrix = flattened_params.reshape((num_particles, -1))
    return matrix, reconstruct_tree


def particle_loss(
    meta_batch_x: Any,
    meta_batch_y: Any,
    prediction_fn: Callable[[chex.ArrayTree, jax.Array], chex.Array],
    particle: models.ParamsMeanField,
    hyper_prior: models.ParamsMeanField,
    key: chex.PRNGKey,
    n_prior_samples: int,
) -> Any:
    """Computes the loss of each SVGD particle of PACOH
    (l. 7, Algorithm 1 PACOH with SVGD approximation of Q*).

    Args:
        meta_batch_x (Array): Input array [meta_batch_dim, batch_dim, input_dim]
        meta_batch_y (Array): Output array [meta_batch_dim, batch_dim, output_dim]
        prediction_fn (Callable[[hk.Params, Array], Array]): Parameterized function
        approximator.
        params (hk.Params): Particle's parameters to learn
        key (PRNGKey): Key for stochasticity.
        n_prior_samples (int): Number of samples.
    Returns:
        Array: Loss.
    """

    def estimate_mll(x: Any, y: Any) -> Any:
        prior_samples = particle.sample(key, n_prior_samples)
        per_sample_pred = jax.vmap(prediction_fn, (0, None))
        y_hat, stddevs = per_sample_pred(prior_samples, x)
        log_likelihood = tfp.distributions.MultivariateNormalDiag(
            y_hat, stddevs
        ).log_prob(y)
        batch_size = x.shape[0]
        mll = jax.scipy.special.logsumexp(
            log_likelihood, axis=0, b=jnp.sqrt(batch_size)
        ) - np.log(n_prior_samples)
        return mll

    # vmap estimate_mll over the task batch dimension, as specified
    # @ Algorithm 1 PACOH with SVGD approximation of Q∗ (MLL_Estimator)
    # @ https://arxiv.org/pdf/2002.05551.pdf.
    mll = jax.vmap(estimate_mll)(meta_batch_x, meta_batch_y)
    log_prob_prior = hyper_prior.log_prob(particle)
    return -(mll + log_prob_prior).mean()


# Based on tf-probability implementation of batched pairwise matrices:
# https://github.com/tensorflow/probability/blob/f3777158691787d3658b5e80883fe1a933d48989/tensorflow_probability/python/math/psd_kernels/internal/util.py#L190
def rbf_kernel(x, y, bandwidth=None):
    """Computes the RBF kernel matrix between (batches of) x and y.
    Returns (batches of) kernel matrices
    :math:`K(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))`.
    """
    row_norm_x = (x**2).sum(-1)[..., None]
    row_norm_y = (y**2).sum(-1)[..., None, :]
    pairwise = jnp.clip(row_norm_x + row_norm_y - 2.0 * jnp.matmul(x, y.T), 0.0)
    n_x = pairwise.shape[-2]
    bandwidth = bandwidth if bandwidth is not None else jnp.median(pairwise)
    bandwidth = 0.5 * bandwidth / jnp.log(n_x + 1)
    bandwidth = jnp.maximum(jax.lax.stop_gradient(bandwidth), 1e-5)
    k_xy = jnp.exp(-pairwise / bandwidth / 2)
    return k_xy


@functools.partial(jax.jit, static_argnums=(3, 5))
def infer_posterior(
    x: chex.Array,
    y: chex.Array,
    hyper_posterior: models.ParamsMeanField,
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
    key: chex.PRNGKey,
    update_steps: int,
    learning_rate: float,
) -> Tuple[chex.ArrayTree, chex.Array]:
    """Infer posterior based on task specific training data.
    The posterior is modeled as an ensemble of neural networks.

    Args:
        x (chex.Array): x-values of task-specific training data.
        [task_dim, batch_dim, input_dim]
        y (chex.Array): y-values of task-specific training data.
        [task_dim, batch_dim, output_dim]
        hyper_posterior (models.ParamsMeanField): Distribution over distributions of
         parameterized functions.
        prediction_fn (Callable[[chex.ArrayTree, chex.Array], chex.Array]):
        parameterizd function.
        key (chex.PRNGKey): PRNG key.
        update_steps (int): Number of update steps to be performed.

    Returns:
        models.ParamsMeanField: Task-inferred posterior.
    """
    # Sample prior parameters from the hyper-posterior to form an ensemble
    # of neural networks.
    posterior_params = hyper_posterior.sample(key, 1)
    posterior_params = jax.tree_util.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), posterior_params
    )
    optimizer = optax.flatten(optax.adam(learning_rate))
    opt_state = optimizer.init(posterior_params)

    def loss(params: hk.Params) -> Any:
        y_hat, stddevs = prediction_fn(params, x)
        log_likelihood = tfp.distributions.MultivariateNormalDiag(
            y_hat, stddevs
        ).log_prob(y)
        return -log_likelihood.mean()

    def update(
        carry: Tuple[chex.ArrayTree, optax.OptState], _: Any
    ) -> Tuple[Tuple[chex.ArrayTree, optax.OptState], chex.Array]:
        posterior_params, opt_state = carry
        values, grads = jax.vmap(jax.value_and_grad(loss))(posterior_params)
        updates, opt_state = optimizer.update(grads, opt_state)
        posterior_params = optax.apply_updates(posterior_params, updates)
        return (posterior_params, opt_state), values.mean()

    (posterior_params, _), losses = jax.lax.scan(
        update, (posterior_params, opt_state), None, update_steps
    )
    return posterior_params, losses


@functools.partial(jax.jit, static_argnums=(2))
def predict(
    posterior: chex.ArrayTree,
    x: chex.Array,
    prediction_fn: Callable[[chex.ArrayTree, chex.Array], chex.Array],
) -> Tuple[chex.Array, chex.Array]:
    """Predict y-values based on the posterior (defined by an ensemble of
     neural networks).
    Args:
        posterior (chex.ArrayTree): Posterior parameters.
        x (chex.Array): x-values of task-specific training data.
        prediction_fn (Callable[[chex.ArrayTree, chex.Array], chex.Array]): Parameterized
        function.

    Returns:
        chex.Array: Prediced mean and standard deviation predicted by each member
        of the ensemble that defines the ensemble.
    """
    prediction_fn = jax.vmap(prediction_fn, in_axes=(0, None))
    y_hat, stddev = prediction_fn(posterior, x)
    return y_hat, stddev


if __name__ == "__main__":
    import models
    import replay_buffer

    class Dataset:
        def __init__(
            self,
            obs,
            action,
            rewards,
            meta_batch_size,
            num_train_shots,
            seed=666,
        ):
            self.buffer = replay_buffer.ReplayBuffer(
                seed,
                obs.shape[-2:],
                action.shape[-2:],
                obs.shape[0],
                obs.shape[1],
                obs.shape[2],
                meta_batch_size,
                num_train_shots,
                99,
                obs,
                action,
                rewards,
            )
            self.meta_batch_size = meta_batch_size
            self.num_train_shots = num_train_shots
            self.rs = np.random.RandomState(seed)

        @property
        def train_set(
            self,
        ):
            while True:
                yield self._make_batch()[0]

        @property
        def test_set(
            self,
        ):
            while True:
                yield self._make_batch()

        def _make_batch(self):
            sample = next(self.buffer.sample(1))
            obs, action, rewards = sample
            # Split each trajectory into test and train set. (yes, the test set is
            # not really a test set, this is cheating...)
            x1 = np.concatenate([obs[:, :, :10], action[:, :, :10]], axis=-1)
            y1 = np.concatenate([obs[:, :, 1:11], rewards[:, :, :10, None]], axis=-1)
            x2 = np.concatenate(
                [
                    obs[:, :, 11:-1],
                    action[:, :, 11:],
                ],
                axis=-1,
            )
            y2 = np.concatenate([obs[:, :, 12:], rewards[:, :, 11:, None]], axis=-1)
            return (x1, y1), (x2, y2)

    obs, action, rewards = np.load("data-50-1-2023-02-13-12:05.npz").values()

    def normalize(x):
        mean = x.mean(axis=(0, 1))
        stddev = x.std(axis=(0, 1))
        return (x - mean) / (stddev + 1e-8), mean, stddev

    obs, *_ = normalize(obs)
    action, *_ = normalize(action)
    rewards, *_ = normalize(rewards)
    # One to take care of the shape of the rewards.
    output_size = (1 + obs.shape[-1],)

    def net(x):
        x = hk.nets.MLP((32, 32, 32, 32) + output_size)(x)
        mu = x
        stddev = hk.get_parameter(
            "stddev", [], init=lambda shape, dtype: jnp.ones(shape, dtype) * 1e-3
        )
        return mu, stddev * jnp.ones_like(mu)

    dataset = Dataset(obs, action, rewards, meta_batch_size=4, num_train_shots=1)
    example = next(dataset.train_set)[0][0]
    init, apply = hk.without_apply_rng(hk.transform(net))
    seed_sequence = hk.PRNGSequence(666)
    mean_prior_over_mus = jax.tree_map(
        jnp.zeros_like, init(next(seed_sequence), example)
    )
    mean_prior_over_stddevs = jax.tree_map(jnp.zeros_like, mean_prior_over_mus)
    hyper_prior = models.ParamsMeanField(
        models.ParamsMeanField(mean_prior_over_mus, mean_prior_over_stddevs),
        0.5,
    )
    n_particles = 10
    init = jax.vmap(init, (0, None))
    particles_mus = init(jnp.asarray(seed_sequence.take(n_particles)), example)
    particle_stddevs = jax.tree_map(lambda x: jnp.ones_like(x) * 1e-4, particles_mus)
    hyper_posterior = models.ParamsMeanField(particles_mus, particle_stddevs)
    infer_posteriors = jax.vmap(
        infer_posterior, in_axes=(0, 0, None, None, None, None, None)
    )
    (context_x, context_y), (test_x, test_y) = next(dataset.test_set)
    posteriors, losses = infer_posteriors(
        context_x, context_y, hyper_posterior, apply, next(seed_sequence), 1000, 3e-4
    )
    predict = jax.vmap(predict, (0, 0, None))
    predictions = predict(posteriors, test_x, apply)
