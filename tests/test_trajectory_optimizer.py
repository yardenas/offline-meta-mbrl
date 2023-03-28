import jax

from sadam.cem import _split, _unsplit


def test_split():
    state = jax.random.normal(jax.random.PRNGKey(0), (4,))
    hidden = [jax.random.normal(jax.random.PRNGKey(1), (128, 64)) for _ in range(2)]
    init_state = _unsplit(hidden, state)
    other_hidden, other_state = _split(
        init_state, state.shape[0], hidden[0].shape, len(hidden)
    )
    assert all(
        jax.tree_map(lambda x, y: jax.numpy.allclose(x, y), hidden, other_hidden)
    )
    assert jax.numpy.allclose(state, other_state)
