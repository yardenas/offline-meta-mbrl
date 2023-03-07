import argparse
from typing import List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax


class SequenceBlock(eqx.Module):
    cell: eqx.nn.GRUCell
    out: eqx.nn.Linear
    out2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    hippo_n: int

    def __init__(self, hidden_size, hippo_n, *, key):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = eqx.nn.GRUCell(hidden_size, hippo_n, key=cell_key)
        self.out = eqx.nn.Linear(hippo_n, hidden_size, key=encoder_key)
        self.out2 = eqx.nn.Linear(hippo_n, hidden_size, key=decoder_key)
        self.norm = eqx.nn.LayerNorm(
            hidden_size,
        )
        self.hippo_n = hippo_n

    def __call__(self, x, convolve=False, *, key=None, ssm=None, hidden=None):
        skip = x
        x = jax.vmap(self.norm)(x)
        hidden = hidden if hidden is not None else jnp.zeros((self.hippo_n,))

        def f(carry, x):
            out = self.cell(x, carry)
            return out, out

        pred_fn = lambda x: jax.lax.scan(f, hidden, x)
        hidden, x = pred_fn(x)
        x = jnn.gelu(x)
        x = jax.vmap(self.out)(x) * jnn.sigmoid(jax.vmap(self.out2)(x))
        return hidden, skip + x


class Model(eqx.Module):
    layers: List[SequenceBlock]
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(self, n_layers, in_size, out_size, hippo_n, hidden_size, *, key):
        keys = jax.random.split(key, n_layers + 2)
        self.layers = [
            SequenceBlock(hidden_size, hippo_n, key=key) for key in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(in_size, hidden_size, key=keys[-2])
        self.decoder = eqx.nn.Linear(hidden_size, out_size, key=keys[-1])

    def __call__(self, x, convolve=False, *, key=None, hidden=None):
        if hidden is None:
            hidden = [None] * len(self.layers)
        x = jax.vmap(self.encoder)(x)
        hidden_states = []
        for layer_hidden, layer in zip(hidden, self.layers):
            hidden, x = layer(x, convolve=convolve, hidden=layer_hidden)
            hidden_states.append(hidden)
        x = jax.vmap(self.decoder)(x)
        if convolve:
            return None, x
        else:
            return hidden_states, x

    def sample(self, horizon, initial_state, inputs=None, *, key=None):
        def f(carry, x):
            i, carry, prev_x = carry
            if x is None:
                x = prev_x
            else:
                prev_x = jnp.concatenate([prev_x[:3], x[-1:]], axis=-1)
                x = jnp.where(i >= horizon, prev_x, x)
            x = x[None]
            out_carry, out = self(x, convolve=False, hidden=carry)
            out = out[0]
            return (i + 1, out_carry, out), out

        _, out = jax.lax.scan(f, (0,) + initial_state, inputs)
        return out

    @property
    def init_state(self):
        return [jnp.zeros((layer.hippo_n,)) for layer in self.layers]


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
    obs, action, reward = [x.squeeze(1) for x in (obs, action, reward)]

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
    data_path="data-25-1-2023-03-06-08:31.npz",
    batch_size=32,
    learning_rate=1e-3,
    steps=500,
    hidden_size=128,
    sequence_length=84,
    seed=777,
):
    loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 2)
    data = get_data(data_path, sequence_length)
    iter_data = dataloader(data, batch_size, key=loader_key)

    model = Model(
        n_layers=4,
        in_size=4,
        out_size=4,
        hidden_size=hidden_size,
        key=model_key,
        hippo_n=64,
    )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        y_hat = jax.vmap(lambda x: model(x, True))(x)[1]
        # Trains with respect to L2 loss
        error = y_hat - y
        return 0.5 * (error**2).mean()

    # Important for efficiency whenever you use JAX: wrap everything
    # into a single JIT region.
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
    hidden = [np.tile(x, (batch_size, 1)) for x in model.init_state]
    y_hat = jax.vmap(model.sample, (None, 0, 0))(1, (hidden, x[:, 0]), x)
    print(f"MSE: {np.mean((y - y_hat)**2)}")
    plot(y, y_hat)


def plot(y, y_hat):
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
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", type=str, required=True)
    # parser.add_argument("--seed", type=int, default=123)
    # args = parser.parse_args()
    main()
