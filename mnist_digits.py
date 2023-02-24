from functools import partial
from typing import List

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

import s4


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jnn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


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
                x, True if x.shape[0] > 64 else False
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

    def __call__(self, x, convolve=False, *, key=None, ssm=None, hidden=None):
        x /= 255.0
        if hidden is None or ssm is None:
            hidden = [None] * len(self.layers)
            ssm = [None] * len(self.layers)
        x = jax.vmap(self.encoder)(x)
        hidden_states = []
        for layer_ssm, layer_hidden, layer in zip(ssm, hidden, self.layers):
            hidden, x = layer(x, convolve=convolve, ssm=layer_ssm, hidden=layer_hidden)
            hidden_states.append(hidden)
        x = jax.vmap(self.decoder)(x)
        x = jnn.log_softmax(x, axis=-1)
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
                x = jnp.where(i >= horizon, prev_x, x)
            x = x[None, :]
            out_carry, out = self(x, convolve=False, ssm=ssms, hidden=carry)
            out = out.argmax(-1).astype(jnp.float32)
            return (i + 1, out_carry, out), out

        _, out = jax.lax.scan(f, (0,) + initial_state, inputs)
        return out


def create_mnist_dataset(bsz=32):
    print("[*] Generating MNIST Sequence Modeling Dataset...")
    # Constants
    SEQ_LENGTH, N_CLASSES, IN_DIM = 784, 256, 1
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x.view(IN_DIM, SEQ_LENGTH).t() * 255).int()),
        ]
    )
    train = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=tf
    )
    test = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=tf
    )
    # Return data loaders, with the provided batch size
    trainloader = data.DataLoader(
        train,
        batch_size=bsz,
        shuffle=True,
    )
    testloader = data.DataLoader(
        test,
        batch_size=bsz,
        shuffle=False,
    )
    return trainloader, testloader, N_CLASSES, IN_DIM


def sample_image_prefix(
    model,
    dataloader,
    prefix=300,
    imshape=(28, 28),
    batch_size=6,
    save=True,
):
    """Sample a grayscale image represented as intensities in [0, 255]"""
    import matplotlib.pyplot as plt
    import numpy as onp

    example = np.array(next(iter(dataloader))[0].numpy())
    batch = example[:batch_size].shape[0]
    length = example.shape[1]
    assert length == onp.prod(imshape)
    it = iter(dataloader)
    image = next(it)[0].numpy().astype(np.float32)
    input_image = np.pad(image[:, :-1], [(0, 0), (1, 0), (0, 0)], constant_values=0)
    curr = input_image[:batch_size]
    _, out1 = jax.vmap(model)(image[:batch_size])
    out1 = out1.argmax(-1)
    hidden = [
        np.tile(layer.cell.init_state, (batch_size, 1, 1)) for layer in model.layers
    ]
    out = jax.vmap(model.sample, (None, 0, 0))(
        prefix, (hidden, jnp.zeros_like(curr[:, 0])), curr
    )
    # Visualization
    if save:
        for k in range(batch):
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
            ax1.set_title("Sampled")
            ax1.imshow(out[k].reshape(imshape) / 256.0)
            ax2.set_title("True")
            ax3.set_title("Single step")
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            ax2.imshow(curr[k].reshape(imshape) / 256.0)
            ax3.imshow(out1[k].reshape(imshape) / 256.0)
            fig.savefig("im.%d.png" % (k))
            plt.close()


def main(
    learning_rate=1e-3,
    steps=4000,
    hidden_size=256,
    seed=777,
):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    trainloader, testloader, n_classes, in_dim = create_mnist_dataset(32)
    iter_data = iter(trainloader)

    model = Model(
        n_layers=4,
        in_size=in_dim,
        out_size=n_classes,
        hidden_size=hidden_size,
        key=model_key,
        hippo_n=64,
    )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        logits = jax.vmap(lambda x: model(x, True))(x)[1].astype(jnp.float32)
        loss = jnp.mean(cross_entropy_loss(logits, y))
        return loss

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
        x, y = [d.numpy() for d in (x, y)]
        y = x[:, 1:, 0]
        x = x[:, :-1]
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    xs, ys = [d.numpy() for d in next(iter(testloader))]
    ys = xs[:, 1:, 0]
    xs = xs[:, :-1]
    logits = jax.vmap(lambda x: model(x, True))(xs)[1]
    final_accuracy = jnp.mean((logits.argmax(-1)) == ys)
    print(f"final_accuracy={final_accuracy}")
    sample_image_prefix(model, testloader)


if __name__ == "__main__":
    main()
