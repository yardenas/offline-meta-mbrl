import jax
import numpy as np

import s4


def test_rnn_cnn(n=8, sequence_length=16):
    u = np.arange(sequence_length) * 1.0
    cell = s4.S4Cell(n, key=jax.random.PRNGKey(666))
    y_cnn = cell.convolve(u)
    ssm = cell.ssm(sequence_length)

    def sequence(x):
        fn = lambda carry, x: cell(carry, x, ssm)
        return jax.lax.scan(fn, np.zeros((n,), dtype=np.complex64), x)

    _, y_rnn = sequence(u)
    y_rnn = y_rnn.squeeze(-1)
    assert np.allclose(y_rnn.real, y_cnn.real, atol=1e-4, rtol=1e-4)


def test_output_dimension_not_equal(n=8, sequence_length=16, input_size=5):
    u = np.ones((sequence_length, input_size))
    cell = s4.S4Cell(n, input_size, key=jax.random.PRNGKey(666))
    y_cnn = cell.convolve(u).real
    sequence_summary = y_cnn.sum(0)
    pairwise_diffs = sequence_summary[:, None] - sequence_summary[None, :]
    # If all the pairwise differences are zero, then the model predicts exactly
    # the same for each dimension in the output
    assert not np.all(np.nonzero(pairwise_diffs))
