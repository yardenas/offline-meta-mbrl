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
            pred_fn = lambda x: jax.lax.scan(fn, self.cell.init_state, x)
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

    def __call__(self, x, convolve=False, *, key=None):
        x = jax.vmap(self.encoder)(x)
        hidden_states = []
        for layer in self.layers:
            hidden, x = layer(x, convolve=convolve)
            hidden_states.append(hidden)
        x = jax.vmap(self.decoder)(x)
        if convolve:
            return None, x
        else:
            return hidden_states, x

    def sample(self, horizon, initial_state, *, key=None):
        ssms = [layer.cell.ssm(horizon) for layer in self.layers]

        def f(carry, x):
            carry, x = carry
            x = self.encoder(x)
            out_carry = []
            for layer_hidden, ssm, layer in zip(carry, ssms, self.layers):
                layer_hidden, x = layer.step(layer_hidden, x, ssm)
                out_carry.append(layer_hidden)
            x = self.decoder(x)
            out = jnp.argmax(x, axis=-1, keepdims=True)
            return (out_carry, out), out

        _, out = jax.lax.scan(f, initial_state, None, horizon)
        return out


def create_mnist_dataset(bsz=128):
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
    final, final2 = None, None
    it = iter(dataloader)
    image = next(it)[0].numpy()
    image = np.pad(image[:, :-1], [(0, 0), (1, 0), (0, 0)], constant_values=0)
    curr = image[:batch_size]
    context = curr[:, :prefix]
    # Cache the first `start` inputs.
    hidden, _ = jax.vmap(model)(context)
    out = jax.vmap(model.sample, (None, 0))(
        curr.shape[1] - context.shape[1], (hidden, context[:, -1])
    )
    # Visualization
    sampled = np.concatenate([context, out], axis=1)
    if save:
        for k in range(batch):
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.set_title("Sampled")
            ax1.imshow(sampled[k].reshape(imshape) / 256.0)
            ax2.set_title("True")
            ax1.axis("off")
            ax2.axis("off")
            ax2.imshow(curr[k].reshape(imshape) / 256.0)
            fig.savefig("im.%d.png" % (k))
            plt.close()
    return final, final2


def main(
    learning_rate=1e-3,
    steps=200,
    hidden_size=128,
    seed=777,
):
    data_key, loader_key, model_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    trainloader, testloader, n_classes, in_dim = create_mnist_dataset()
    iter_data = iter(trainloader)

    model = Model(
        n_layers=1,
        in_size=in_dim,
        out_size=n_classes,
        hidden_size=hidden_size,
        key=model_key,
        hippo_n=64,
    )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y):
        logits = jax.vmap(lambda x: model(x, True))(x)[1].astype(jnp.float32)
        loss = jax.vmap(jax.vmap(optax.softmax_cross_entropy))(
            logits, jnn.one_hot(y, n_classes)
        ).mean()
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
        grads = eqx.tree_at(cells, grads, replace_fn=lambda x: x * 1.0)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    optim = optax.adam(learning_rate)
    opt_state = optim.init(model)
    for step, (x, y) in zip(range(steps), iter_data):
        x, y = [d.numpy() for d in (x, y)]
        y = x[:, :-1, 0]
        x = x[:, 1:]
        loss, model, opt_state = make_step(model, x, y, opt_state)
        loss = loss.item()
        print(f"step={step}, loss={loss}")

    xs, ys = [d.numpy() for d in next(iter(testloader))]
    ys = xs[:, :-1, 0]
    xs = xs[:, 1:]
    logits = jax.vmap(lambda x: model(x, True))(xs)[1]
    num_correct = jnp.sum((jnp.exp(logits).argmax(-1)) == ys)
    final_accuracy = (num_correct / xs.size).item()
    print(f"final_accuracy={final_accuracy}")

    img1, img2 = sample_image_prefix(model, testloader)


if __name__ == "__main__":
    main()
