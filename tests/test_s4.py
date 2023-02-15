import jax
import numpy as np

import s4


def test_rnn_cnn(n=8, sequence_length=16):
    u = np.arange(sequence_length) * 1.0
    cell = s4.S4Cell(n, sequence_length, key=jax.random.PRNGKey(666))
    y_cnn = cell.multistep(u)
    ssm = cell.ssm

    def sequence(x):
        fn = lambda carry, x: cell(carry, x, ssm)
        return jax.lax.scan(fn, np.zeros((n,), dtype=np.complex64), x)

    _, y_rnn = sequence(u)
    y_rnn = y_rnn.squeeze(-1)
    assert np.allclose(y_rnn.real, y_cnn.real, atol=1e-4, rtol=1e-4)
