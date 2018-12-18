from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from Neuron import Neuron


def sine(theta, phase, radius=1, bias=0):
    return radius * np.sin((theta + phase) * 2 * np.pi) + bias


def gen(tn, tau, func):
    return np.array([f(tn / tau) for f in func]).reshape(-1, 1)


def show(filename, data):
    # Present Data
    cornflower = (.3, .6, .9, .5)
    cornflower_transparent = (.3, .6, .9, .1)
    dark_red = (0.75, 0, 0)
    dark_green = (0, 0.75, 0)

    ylabel = ["${}$".format(n) for n in "uxyz"]
    lw = [1 for _ in range(len(data))]
    colors = [dark_red, None, dark_green, None]
    alpha = [1 for _ in range(len(data))]
    plot_gap = 2.2

    # VISUALIZATION
    fig, ax = plt.subplots(1, len(data), sharex="all", figsize=(5 * len(data), 5))
    for i in range(len(data)):
        for j in range(data[i].shape[1]):
            bias = plot_gap * j
            data[i][:, j] += bias
            ax[i].axhline(y=bias, lw=0.5, alpha=0.5)
            if data[i].shape[1] <= 20:
                ax[i].axhline(y=bias + plot_gap / 2, lw=1, color=cornflower)
                ax[i].axhline(y=bias - plot_gap / 2, lw=1, color=cornflower)
        ax[i].spines['top'].set_color(cornflower)
        ax[i].spines['bottom'].set_color(cornflower)
        ax[i].spines['left'].set_color(cornflower)
        ax[i].spines['right'].set_color(cornflower)
        ax[i].patch.set_facecolor(cornflower_transparent)
        if i + 1 < len(data) and i > 0:
            ax[i].set_yticks([])
        ax[i].plot(data[i], lw=lw[i], color=colors[i], alpha=alpha[i])
        ax[i].set_title("ESN Activations : {}".format(["Input", "Reservoir", "Output", "State"][i]))
        ax[i].set_xlabel("time steps")
        ax[i].set_ylabel("{} has {} channels".format(ylabel[i], data[i].shape[1]))

    fig.savefig(filename)
    plt.close(fig)


def load_model(loadExisting=False):
    neuron = Neuron.load() if loadExisting else None
    if neuron is None:
        neuron = Neuron(
            numInputs=1,
            numReservoir=5,
            numOutputs=1,
            pct=1.0,
            alpha=0.9,
            f=Neuron.tanh,
            g=Neuron.tanh,
            feedback=False)
        neuron.save()
    return neuron


def evaluate(neuron, fname):
    cnt = 500

    neuron.settle(100)

    idx = 0
    # Setup local variables to hold all return values
    u = np.zeros((cnt, neuron.numInputs))
    x = np.zeros((cnt, neuron.numReservoir))
    y = np.zeros((cnt, neuron.numOutputs))
    z = np.zeros((cnt, neuron.numReservoir + neuron.numInputs))

    # Generate the entry passed to the ESN - a funcion.  It implements f(t_n) = r*sin((theta+phase)*2pi)
    f_in = partial(gen, tau=(cnt - 1) / 2,
                   func=[partial(sine, phase=x, radius=0, bias=0.0) for x in
                         np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    # Run Network
    for n in range(cnt):
        zzz = neuron.cycle(n, n, f_in)
        u[n] = zzz[0].reshape(zzz[0].shape[0])
        x[n] = zzz[1].reshape(zzz[1].shape[0])
        y[n] = zzz[2].reshape(zzz[2].shape[0])
        z[n] = zzz[3].reshape(zzz[3].shape[0])
    show(fname.format(idx), [u, x, y])
    idx += 1

    neuron.settle(100)

    # Generate the entry passed to the ESN - a funcion.  It implements f(t_n) = r*sin((theta+phase)*2pi)
    f_in = partial(gen, tau=(cnt - 1) / 2,
                   func=[partial(sine, phase=x, radius=0, bias=0.5) for x in
                         np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    # Run Network
    for n in range(cnt):
        zzz = neuron.cycle(n, n, f_in)
        u[n] = zzz[0].reshape(zzz[0].shape[0])
        x[n] = zzz[1].reshape(zzz[1].shape[0])
        y[n] = zzz[2].reshape(zzz[2].shape[0])
        z[n] = zzz[3].reshape(zzz[3].shape[0])
    show(fname.format(idx), [u, x, y])
    idx += 1


def train(neuron):
    cnt = 500

    # Generate the entry passed to the ESN - a funcion.  It implements f(t_n) = r*sin((theta+phase)*2pi)
    f_in1 = partial(gen, tau=(cnt - 1) / 2,
                    func=[partial(sine, phase=x, radius=0, bias=0.0) for x in
                          np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    f_in2 = partial(gen, tau=(cnt - 1) / 2,
                    func=[partial(sine, phase=x, radius=0, bias=0.5) for x in
                          np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    f_in3 = partial(gen, tau=(cnt - 1) / 2,
                    func=[partial(sine, phase=x, radius=0, bias=-0.5) for x in
                          np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    f_in4 = partial(gen, tau=(cnt - 1) / 2,
                    func=[partial(sine, phase=x, radius=0, bias=0.0) for x in
                          np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    def f_in(t):
        if t < cnt * 1 / 4:
            return f_in1(t)
        elif t < cnt * 2 / 4:
            return f_in2(t)
        elif t < cnt * 3 / 4:
            return f_in3(t)
        else:  # t < cnt * 4 / 4:
            return f_in4(t)

    # Setup local variables to hold all return values
    u = np.zeros((cnt, neuron.numInputs))
    x = np.zeros((cnt, neuron.numReservoir))
    y = np.zeros((cnt, neuron.numOutputs))
    z = np.zeros((cnt, neuron.numReservoir + neuron.numInputs))

    # Run Network
    for n in range(cnt):
        zzz = neuron.cycle(n, n, f_in)
        u[n] = zzz[0].reshape(zzz[0].shape[0])
        x[n] = zzz[1].reshape(zzz[1].shape[0])
        y[n] = zzz[2].reshape(zzz[2].shape[0])
        z[n] = zzz[3].reshape(zzz[3].shape[0])
    show("esn0_VisualizeTrainingInput.png", [u, x, y])

    #################################################################################################################
    # Train Network
    #################################################################################################################
    neuron.forget()

    # Generate the entry passed to the ESN - a funcion.  It implements f(t_n) = r*sin((theta+phase)*2pi)
    f_out = partial(gen, tau=(cnt - 1) / 2,
                    func=[partial(sine, phase=x, radius=1.0, bias=0) for x in
                          np.linspace(0, 1, neuron.numInputs, endpoint=True)])

    neuron.force_learn(0, cnt - 1, f_in, f_out)


def main():
    neuron = load_model()

    evaluate(neuron, "esn0_Start{}.png")
    train(neuron)
    evaluate(neuron, "esn0_End{}.png")


if __name__ == "__main__":
    np.seterr(all="raise")
    np.set_printoptions(precision=4, linewidth=512, floatmode='fixed', suppress=True, sign=' ')

    main()
