from typing import List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax

import s4


class SequenceBlock(eqx.Module):
    cell: s4.S4Cell
    out: eqx.nn.Linear
    out2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(self, hidden_size, hippo_n, *, key):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = s4.S4Cell(hippo_n, hidden_size, key=cell_key)
        self.out = eqx.nn.Linear(hidden_size, hidden_size, key=encoder_key)
        self.out2 = eqx.nn.Linear(hidden_size, hidden_size, key=decoder_key)
        self.norm = eqx.nn.LayerNorm(
            hidden_size,
        )

    def __call__(self, x, convolve=False, *, key=None):
        skip = x
        x = jax.vmap(self.norm)(x)
        if convolve:
            # Heuristically use FFT for very long sequence lengthes
            pred_fn = lambda x: self.cell.convolve(
                x, True if x.shape[0] > 64 else False
            )
        else:
            fn = lambda carry, x: self.cell(carry, x, self.cell.ssm(skip.shape[0]))
            pred_fn = lambda x: jax.lax.scan(fn, self.cell.init_state, x)[1]
        x = jnn.gelu(pred_fn(x))
        x = jax.vmap(self.out)(x) * jnn.sigmoid(jax.vmap(self.out2)(x))
        return skip + x

    def step(self, hidden, x, ssm):
        skip = x
        x = self.norm(x)
        hidden, x = self.cell(hidden, x, ssm)
        x = jnn.gelu(x)
        x = self.out(x) * jnn.sigmoid(self.out2(x))
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

    def __call__(self, x, convolve=False, *, key=None):
        x = jax.vmap(self.encoder)(x)
        for layer in self.layers:
            x = layer(x, convolve=convolve)
        x = jax.vmap(self.decoder)(x)
        return x

    def sample(self, initial_state, action_sequence, *, key=None):
        ssms = [layer.cell.ssm(action_sequence.shape[0]) for layer in self.layers]

        def f(carry, x):
            state, carry = carry
            x = jnp.concatenate([state, x], -1)
            x = self.encoder(x)
            out_carry = []
            for layer_hidden, ssm, layer in zip(carry, ssms, self.layers):
                layer_hidden, x = layer.step(layer_hidden, x, ssm)
                out_carry.append(layer_hidden)
            x = self.decoder(x)
            return (x[:-1], out_carry), x

        init = initial_state, [layer.cell.init_state for layer in self.layers]
        _, out = jax.lax.scan(f, init, action_sequence)
        return out


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
    obs, action, reward = [x.squeeze(0) for x in (obs, action, reward)]

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
    data_path="data-1-25-2023-02-15-15:46.npz",
    batch_size=32,
    learning_rate=1e-3,
    steps=200,
    hidden_size=128,
    sequence_length=16,
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
        y_hat = jax.vmap(lambda x: model(x, True))(x)
        # Trains with respect to L2 loss
        error = y_hat - y
        return 0.5 * (error**2).mean()

    # Important for efficiency whenever you use JAX: wrap everything
    # into a single JIT region.
    def cells(grads):
        nodes = []
        for layer in grads.layers:
            nodes.append(layer.cell.lambda_real)
            nodes.append(layer.cell.lambda_imag)
            nodes.append(layer.cell.p)
            nodes.append(layer.cell.b)
            nodes.append(layer.cell.c)
            nodes.append(layer.cell.d)
        return nodes

    @eqx.filter_jit
    def make_step(model, x, y, opt_state):
        loss, grads = compute_loss(model, x, y)
        grads = eqx.tree_at(cells, grads, replace_fn=lambda x: x * 1.0)
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
    initial_state = x[:, 0, :-1]
    action_sequence = x[..., -1:]
    jax.vmap(model.sample)(initial_state, action_sequence)


if __name__ == "__main__":
    main()
