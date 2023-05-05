from typing import List, NamedTuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax


def to_ins(observation, action):
    return jnp.concatenate([observation, action], -1)


def to_outs(next_state, reward):
    return jnp.concatenate([next_state, reward[..., None]], -1)


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array


class FeedForwardModel(eqx.Module):
    layers: list[eqx.nn.Linear]
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    stddev: jax.Array

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 2 + n_layers)
        self.layers = [
            eqx.nn.Linear(hidden_size, hidden_size, key=k) for k in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(state_dim + action_dim, hidden_size, key=keys[-2])
        self.decoder = eqx.nn.Linear(hidden_size, state_dim + 1, key=keys[-1])
        self.stddev = jnp.ones((state_dim + 1,)) * -10.0

    def __call__(self, x: jax.Array):
        x = jax.vmap(self.encoder)(x)
        for layer in self.layers:
            x = jnn.relu(jax.vmap(layer)(x))
        split = lambda x: jnp.split(x, [self.decoder.out_features - 1], -1)  # type: ignore # noqa E501
        pred = jax.vmap(self.decoder)(x)
        state, reward = split(pred)
        return to_outs(state, reward.squeeze(-1))

    def step(self, state: jax.Array, action: jax.Array) -> Prediction:
        batched = True
        if state.ndim == 1:
            state, action = state[None], action[None]
            batched = False
        x = to_ins(state, action)
        mus = self(x)
        if not batched:
            mus = mus.squeeze(0)
        split = lambda x: jnp.split(x, [self.decoder.out_features - 1], -1)  # type: ignore # noqa E501
        state_mu, reward_mu = split(mus)
        reward_mu = reward_mu.squeeze(-1)
        return Prediction(state_mu, reward_mu)

    def sample(
        self,
        horizon: int,
        initial_state: jax.Array,
        key: jax.random.KeyArray,
        action_sequence: jax.Array,
    ) -> Prediction:
        def f(carry, x):
            prev_state = carry
            action, key = x
            out = self.step(
                prev_state,
                action,
            )
            return out.next_state, out

        init = initial_state
        inputs = (action_sequence, jax.random.split(key, horizon))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out  # type: ignore


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jax.random.permutation(key, indices)
        (key,) = jax.random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            obs, next_obs, acs, rews = tuple(array[batch_perm] for array in arrays)
            yield np.concatenate((obs, acs), -1), np.concatenate(
                (next_obs, rews[..., None]), -1
            )
            start = end
            end = start + batch_size


def get_data(data_path, sequence_length):
    obs, action, reward = np.load(data_path).values()
    obs, action, reward = [x.reshape(-1, *x.shape[2:]) for x in (obs, action, reward)]

    def normalize(x):
        mean = x.mean(axis=(0))
        stddev = x.std(axis=(0))
        return (x - mean) / (stddev + 1e-8), mean, stddev

    obs, *_ = normalize(obs)
    action, *_ = normalize(action)
    reward, *_ = normalize(reward)
    all_obs, all_next_obs, all_acs, all_rews = [], [], [], []
    for t in range(action.shape[1] - sequence_length):
        all_obs.append(obs[:, t : t + sequence_length])
        all_next_obs.append(obs[:, t + 1 : t + sequence_length + 1])
        all_rews.append(reward[:, t : t + sequence_length])
        all_acs.append(action[:, t : t + sequence_length])
    obs, next_obs, acs, rews = map(
        lambda x: np.concatenate(x, axis=0), (all_obs, all_next_obs, all_acs, all_rews)  # type: ignore # noqa: E501
    )
    return obs, next_obs, acs, rews


def main(
    data_path="data-200-multi.npz",
    batch_size=32,
    learning_rate=3e-4,
    steps=500,
    hidden_size=128,
    sequence_length=84,
    seed=777,
):
    loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    data = get_data(data_path, sequence_length)
    iter_data = dataloader(data, batch_size, key=loader_key)

    model = FeedForwardModel(
        n_layers=3,
        state_dim=3,
        action_dim=1,
        hidden_size=hidden_size,
        key=model_key,
    )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        y_hat = jax.vmap(model)(x)
        # Trains with respect to L2 loss
        error = y_hat - y
        return 0.5 * (error**2).mean()

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")
    x, y = next(iter_data)
    context = 25
    y_hat = jax.vmap(model.sample, (None, 0, None, 0))(
        sequence_length,
        x[:, 0, ..., :-1],
        jax.random.PRNGKey(0),
        x[:, ..., -1:],
    )
    y_hat = to_outs(y_hat.next_state, y_hat.reward)
    print(f"MSE: {np.mean((y - y_hat)**2)}")
    plot(y, y_hat, context)


def plot(y, y_hat, context):
    import matplotlib.pyplot as plt

    t = np.arange(y.shape[1])

    plt.figure(figsize=(10, 5), dpi=600)
    for i in range(4):
        plt.subplot(2, 3, i + 1)
        plt.plot(t, y[i, :, 2], "b.", label="observed")
        plt.plot(
            t,
            y_hat[i, :, 2],
            "r",
            label="prediction",
            linewidth=1.0,
        )
        ax = plt.gca()
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(("data", 0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvline(context, color="k", linestyle="--", linewidth=1.0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
