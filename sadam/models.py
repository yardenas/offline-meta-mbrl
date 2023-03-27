from typing import List, NamedTuple

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp

import sadam.s4 as s4


class Prediction(NamedTuple):
    next_state: jax.Array
    reward: jax.Array


class SequenceBlock(eqx.Module):
    cell: s4.S4Cell
    out: eqx.nn.Linear
    out2: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    hippo_n: int

    def __init__(self, hidden_size, hippo_n, sequence_length, *, key):
        cell_key, encoder_key, decoder_key = jax.random.split(key, 3)
        self.cell = s4.S4Cell(hippo_n, hidden_size, sequence_length, key=cell_key)
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
            ssm = ssm if ssm is not None else self.cell.ssm
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
    state_decoder: eqx.nn.Linear
    reward_decoder: eqx.nn.Linear

    def __init__(
        self, n_layers, in_size, out_size, hippo_n, hidden_size, sequence_length, *, key
    ):
        keys = jax.random.split(key, n_layers + 2)
        self.layers = [
            SequenceBlock(hidden_size, hippo_n, sequence_length, key=key)
            for key in keys[:n_layers]
        ]
        self.encoder = eqx.nn.Linear(in_size, hidden_size, key=keys[-2])
        self.state_decoder = eqx.nn.Linear(hidden_size, out_size, key=keys[-1])
        self.reward_decoder = eqx.nn.Linear(hidden_size, out_size, key=keys[-1])

    def __call__(
        self,
        state_sequence,
        action_sequence,
        convolve=False,
        *,
        key=None,
        ssm=None,
        hidden=None
    ):
        if hidden is None or ssm is None:
            hidden = [None] * len(self.layers)
            ssm = [None] * len(self.layers)
        x = jax.vmap(self.encoder)(
            jnp.concatenate([state_sequence, action_sequence], -1)
        )
        hidden_states = []
        for layer_ssm, layer_hidden, layer in zip(ssm, hidden, self.layers):
            hidden, x = layer(x, convolve=convolve, ssm=layer_ssm, hidden=layer_hidden)
            hidden_states.append(hidden)
        next_state = jax.vmap(self.state_decoder)(x)
        reward = jax.vmap(self.reward_decoder)(x)
        if convolve:
            return None, Prediction(next_state, reward)
        else:
            return hidden_states, Prediction(next_state, reward)

    def sample(
        self,
        horizon,
        initial_state,
        initial_hidden,
        action_sequence=None,
        policy=None,
        ssm=None,
        *,
        key
    ):
        def f(carry, x):
            prev_hidden, prev_state = carry
            action, key = x
            if action is None:
                assert policy is not None
                action = policy(prev_state).sample(key)
            out_hidden, out = self(
                prev_state, action, convolve=False, ssm=ssm, hidden=prev_hidden
            )
            return (out_hidden, out.next_state), out

        if ssm is None:
            ssm = [layer.cell.ssm for layer in self.layers]
        if action_sequence is None:
            assert action_sequence is not None
            action_sequence = [None] * len(horizon)
        init = (initial_hidden, initial_state)
        inputs = (action_sequence, jax.random.split(key, len(horizon)))
        _, out = jax.lax.scan(
            f,
            init,
            inputs,
        )
        return out

    @property
    def init_state(self):
        return [layer.cell.init_state for layer in self.layers]
