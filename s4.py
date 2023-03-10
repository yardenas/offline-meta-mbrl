import equinox as eqx
import jax
import jax.numpy as jnp


def make_HiPPO(n):
    p = jnp.sqrt(1 + 2 * jnp.arange(n))
    a = p[:, jnp.newaxis] * p[jnp.newaxis, :]
    a = jnp.tril(a) - jnp.diag(jnp.arange(n))
    return -a


def make_NPLR_HiPPO(n):
    # Make -HiPPO
    nhippo = make_HiPPO(n)

    # Add in a rank 1 term. Makes it Normal.
    p = jnp.sqrt(jnp.arange(n) + 0.5)

    # HiPPO also specifies the B matrix
    b = jnp.sqrt(2 * jnp.arange(n) + 1.0)
    return nhippo, p, b


def make_DPLR_HiPPO(n):
    """Diagonalize NPLR representation"""
    a, p, b = make_NPLR_HiPPO(n)

    s = a + p[:, jnp.newaxis] * p[jnp.newaxis, :]

    # Check skew symmetry
    s_diag = jnp.diagonal(s)
    lambda_real = jnp.mean(s_diag) * jnp.ones_like(s_diag)

    # Diagonalize S to V \lambda V^*
    lambda_imag, v = jnp.linalg.eigh(s * -1j)

    p = v.conj().T @ p
    b = v.conj().T @ b
    return lambda_real + 1j * lambda_imag, p, b, v


def hippo_initializer(n):
    _lambda, p, b, _ = make_DPLR_HiPPO(n)
    return _lambda.real, _lambda.imag, p, b


def log_step_initializer(key, shape, dt_min=0.001, dt_max=0.1):
    return jax.random.uniform(key, shape) * (
        jnp.log(dt_max) - jnp.log(dt_min)
    ) + jnp.log(dt_min)


def discrete_DPLR(_lambda, p, q, b, c, step, sequence_length):
    # Convert parameters to matrices
    b = b[:, None]
    ct = c[None, :]

    n = _lambda.shape[0]
    a = jnp.diag(_lambda) - p[:, None] @ q[:, None].conj().T
    i = jnp.eye(n)

    # Forward Euler
    a0 = (2.0 / step) * i + a

    # Backward Euler
    d = jnp.diag(1.0 / ((2.0 / step) - _lambda))
    qc = q.conj().T.reshape(1, -1)
    p2 = p.reshape(-1, 1)
    a1 = d - (d @ p2 * (1.0 / (1 + (qc @ d @ p2))) * qc @ d)

    # A bar and B bar
    ab = a1 @ a0
    bb = 2 * a1 @ b

    # Recover Cbar from Ct
    cb = ct @ jnp.linalg.inv(i - jnp.linalg.matrix_power(ab, sequence_length)).conj()
    return ab, bb, cb.conj()


def cauchy(v, omega, lambd):
    """Cauchy matrix multiplication: (n), (l), (n) -> (l)"""
    cauchy_dot = lambda _omega: (v / (_omega - lambd)).sum()
    return jax.vmap(cauchy_dot)(omega)


def kernel_DPLR(_lambda, p, q, b, c, step, sequence_length):
    # Evaluate at roots of unity
    # Generating function is (-)z-transform, so we evaluate at (-)root
    omega_l = jnp.exp((-2j * jnp.pi) * (jnp.arange(sequence_length) / sequence_length))
    aterm = (c.conj(), q.conj())
    bterm = (b, p)

    g = (2.0 / step) * ((1.0 - omega_l) / (1.0 + omega_l))
    c = 2.0 / (1.0 + omega_l)

    # Reduction to core Cauchy kernel
    k00 = cauchy(aterm[0] * bterm[0], g, _lambda)
    k01 = cauchy(aterm[0] * bterm[1], g, _lambda)
    k10 = cauchy(aterm[1] * bterm[0], g, _lambda)
    k11 = cauchy(aterm[1] * bterm[1], g, _lambda)
    roots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
    out = jnp.fft.ifft(roots, sequence_length).reshape(sequence_length)
    return out.real


def _convolve(_lambda, p, b, c, d, step, u, fft):
    k = kernel_DPLR(
        _lambda,
        p,
        p,
        b,
        c,
        step,
        u.shape[0],
    )
    if fft:
        ud = jnp.fft.rfft(jnp.pad(u, (0, k.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(k, (0, u.shape[0])))
        out = ud * Kd
        out = jnp.fft.irfft(out)[: u.shape[0]]
    else:
        out = jnp.convolve(u, k, mode="full")[: u.shape[0]]
    return (out + d * u).real


class S4Cell(eqx.Module):
    lambda_real: jnp.ndarray
    lambda_imag: jnp.ndarray
    p: jnp.ndarray
    b: jnp.ndarray
    c: jnp.ndarray
    d: jnp.ndarray
    step: jnp.ndarray

    def __init__(self, hippo_n, input_size, *, key):
        hippo_params = hippo_initializer(hippo_n)
        _lambda_real, _lambda_imag, p, b = [
            jnp.tile(x, (input_size, 1)) for x in hippo_params
        ]
        self.lambda_real = _lambda_real
        self.lambda_imag = _lambda_imag
        self.p = p
        self.b = b
        self.c = jax.random.normal(key, (input_size, hippo_n, 2)) * (0.5**0.5)
        self.d = jnp.ones((input_size, 1))
        key, _ = jax.random.split(key)
        self.step = log_step_initializer(
            key,
            (
                input_size,
                1,
            ),
        )

    @jax.vmap
    def __call__(self, x_k_1, u_k, ssm):
        ab, bb, cb = ssm
        if u_k.ndim == 0:
            u_k = u_k[None]
        x_k = ab @ x_k_1 + bb @ u_k
        y_k = cb @ x_k
        return x_k, (y_k + self.d * u_k).real.squeeze(-1)

    def convolve(self, u, fft=False):
        return jax.vmap(_convolve, (0,) * 6 + (1, None), 1)(
            jnp.clip(self.lambda_real, None, -1e-4) + 1j * self.lambda_imag,
            self.p,
            self.b,
            self.c[..., 0] + 1j * self.c[..., 1],
            self.d,
            jnp.exp(self.step),
            u,
            fft,
        )

    def ssm(self, sequence_length):
        return jax.vmap(discrete_DPLR, in_axes=(0,) * 6 + (None,))(
            jnp.clip(self.lambda_real, None, -1e-4) + 1j * self.lambda_imag,
            self.p,
            self.p,
            self.b,
            self.c[..., 0] + 1j * self.c[..., 1],
            jnp.exp(self.step),
            sequence_length,
        )

    @property
    def init_state(self):
        return jnp.zeros_like(self.b)
