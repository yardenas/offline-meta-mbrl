from typing import List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

import sadam.s4 as s4


class SequenceBlock(eqx.Module):
    cell: s4.S4Cell
    out: eqx.nn.Linear
    out2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    hippo_n: int

    def __init__(self, hidden_size, hippo_n, *, key):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = s4.S4Cell(hippo_n, hidden_size, key=cell_key)
        self.out = eqx.nn.Linear(hidden_size, hidden_size, key=encoder_key)
        self.out2 = eqx.nn.Linear(hidden_size, hidden_size, key=decoder_key)
        self.norm = eqx.nn.LayerNorm(
            hidden_size,
        )
        self.hippo_n = hippo_n

    def __call__(self, x, convolve=False, *, key=None, ssm=None, hidden=None):
        skip = x
        x = jax.vmap(self.norm)(x)
        if convolve:
            # Heuristically use FFT for very long sequence lengthes
            pred_fn = lambda x: self.cell.convolve(
                x, True if x.shape[0] > 32 else False
            )
        else:
            ssm = ssm if ssm is not None else self.cell.ssm(skip.shape[0])
            hidden = hidden if hidden is not None else self.cell.init_state
            fn = lambda carry, x: self.cell(carry, x, ssm)
            pred_fn = lambda x: jax.lax.scan(fn, hidden, x)
        if convolve:
            x = pred_fn(x)
            hidden = None
        else:
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

    def __call__(self, x, convolve=False, *, key=None, ssm=None, hidden=None):
        if hidden is None or ssm is None:
            hidden = [None] * len(self.layers)
            ssm = [None] * len(self.layers)
        x = jax.vmap(self.encoder)(x)
        hidden_states = []
        for layer_ssm, layer_hidden, layer in zip(ssm, hidden, self.layers):
            hidden, x = layer(x, convolve=convolve, ssm=layer_ssm, hidden=layer_hidden)
            hidden_states.append(hidden)
        x = jax.vmap(self.decoder)(x)
        if convolve:
            return None, x
        else:
            return hidden_states, x

    def sample(self, horizon, initial_state, inputs=None, *, key=None):
        sequence_length = horizon if inputs is None else inputs.shape[0]
        ssms = [layer.cell.ssm(sequence_length) for layer in self.layers]

        def f(carry, x):
            i, carry, prev_x = carry
            if x is None:
                x = prev_x
            else:
                prev_x = jnp.concatenate([prev_x[:3], x[-1:]], axis=-1)
                x = jnp.where(i >= horizon, prev_x, x)
            x = x[None]  # pyright: ignore
            out_carry, out = self(x, convolve=False, ssm=ssms, hidden=carry)
            out = out[0]
            return (i + 1, out_carry, out), out

        _, out = jax.lax.scan(f, (0,) + initial_state, inputs)
        return out

    @property
    def init_state(self):
        return [layer.cell.init_state for layer in self.layers]
