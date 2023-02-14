from dataclasses import asdict

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
    hippo_n: int

    def __init__(self, hidden_size, hippo_n, sequence_length, *, key):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = s4.S4Cell(hippo_n, sequence_length, key=cell_key)
        # self.cell = eqx.nn.GRUCell(hidden_size, hidden_size, key=cell_key)
        self.out = eqx.nn.Linear(hidden_size, hidden_size, key=encoder_key)
        self.out2 = eqx.nn.Linear(hidden_size, hidden_size, key=decoder_key)
        self.hippo_n = hippo_n

    def __call__(self, x, *, key=None):
        skip = x
        x0 = jnp.zeros((self.hippo_n,), jnp.complex64)
        ssm = self.cell.ssm

        def sequence(x):
            fn = lambda carry, x: self.cell(carry, x, ssm)
            return jax.lax.scan(fn, x0, x)

        _, out = jax.vmap(sequence)(x)
        x = jnn.gelu(out.squeeze())
        x = jax.vmap(self.out)(x) * jnn.sigmoid(jax.vmap(self.out2)(x))
        return skip + x

    # def __call__(self, x, *, key=None):
    #     skip = x
    #     x0 = jnp.zeros((64,), jnp.complex64)

    #     def f(carry, inp):
    #         out = self.cell(inp, carry)
    #         return out, out

    #     _, out = jax.lax.scan(f, x0, x)
    #     x = jnn.gelu(out)
    #     x = jax.vmap(self.out)(x) * jnn.sigmoid(jax.vmap(self.out2)(x))
    #     return skip + x


class Model(eqx.Module):
    layers: eqx.nn.Sequential
    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear

    def __init__(
        self, n_layers, in_size, out_size, hippo_n, sequence_length, hidden_size, *, key
    ):
        keys = jax.random.split(key, n_layers + 2)
        self.layers = eqx.nn.Sequential(
            [
                SequenceBlock(hidden_size, hippo_n, sequence_length, key=key)
                for key in keys[:n_layers]
            ]
        )
        self.encoder = eqx.nn.Linear(in_size, hidden_size, key=keys[-2])
        self.decoder = eqx.nn.Linear(hidden_size, out_size, key=keys[-1])

    def __call__(self, x):
        x = jnn.relu(jax.vmap(self.encoder)(x))
        x = self.layers(x)
        x = x.mean(axis=0)
        x = self.decoder(x)
        return jnn.sigmoid(x)


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.nn.GRUCell
    linear: eqx.nn.Linear
    bias: jnp.ndarray

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jax.random.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=False, key=lkey)
        self.bias = jnp.zeros(out_size)

    def __call__(self, input):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            return self.cell(inp, carry), None

        out, _ = jax.lax.scan(f, hidden, input)
        # sigmoid because we're performing binary classification
        return jax.nn.sigmoid(self.linear(out) + self.bias)


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
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def get_data(dataset_size, sequence_length, *, key):
    t = jnp.linspace(0, 2 * np.pi, sequence_length)
    offset = jax.random.uniform(key, (dataset_size, 1), minval=0, maxval=2 * np.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return x, y


def map_nested_fn(fn):
    """Recursively apply `fn` to the key-value pairs of a nested dict"""

    def map_fn(nested_dict):
        if not isinstance(nested_dict, dict):
            nested_dict = asdict(nested_dict)
        return {
            k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def label_params(key, value):
    if key in ["p", "b", "c", "d", "lambda_real", "lambda_image"]:
        return "Yes"
    return "No"


def main(
    dataset_size=10000,
    batch_size=32,
    learning_rate=3e-3,
    steps=200,
    hidden_size=64,
    sequence_length=16,
    seed=5678,
):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    xs, ys = get_data(dataset_size, sequence_length, key=data_key)
    iter_data = dataloader((xs, ys), batch_size, key=loader_key)

    model = Model(
        n_layers=1,
        in_size=2,
        out_size=1,
        hidden_size=hidden_size,
        key=model_key,
        hippo_n=32,
        sequence_length=16,
    )
    # model = RNN(in_size=2, out_size=1, hidden_size=hidden_size, key=model_key)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        pred_y = jax.vmap(model)(x).astype(jnp.float32)
        # Trains with respect to binary cross-entropy
        return -jnp.mean(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))

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
        grads = eqx.tree_at(cells, grads, replace_fn=lambda x: x * 10.0)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    pred_ys = jax.vmap(model)(xs)
    num_correct = jnp.sum((pred_ys > 0.5) == ys)
    final_accuracy = (num_correct / dataset_size).item()
    print(f"final_accuracy={final_accuracy}")


if __name__ == "__main__":
    main()
