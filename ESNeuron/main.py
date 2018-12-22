import sys
from functools import partial
from typing import TypeVar, Sequence, List
import matplotlib.pyplot as plt
import numpy as np

from Neuron import *

T = TypeVar("T")


def sine(theta: T, phase=0, radius=1, bias=0) -> T:
    return radius * np.sin((theta + phase) * 2 * np.pi) + bias


def iden(bias: T) -> T:
    return bias


def gen(tn: float, tau: float, func: Sequence[Callback]) -> np.ndarray:
    """
    Execute functions f from <func> over input <tn> with time constant <tau>.  Return the results in a column matrix
    :param tn: The numerator of the parameter to pass to each function f in func
    :param tau: The denominator of the parameter to pass to each function f in func
    :param func: A list like structure of functions f to execute
    :return: A column vector of function call results
    """
    return np.array([f(tn / tau) for f in func]).reshape(-1, 1)

class Callback:
    def __init__(self, tau, func, phase, radius, bias, channels):
        self._tau = tau
        self._func = func
        self._phase = phase
        self._radius = radius
        self._bias = bias
        self._channels = channels

    def __call__(self, theta):
        return self._func(theta, phase=self._phase, radius=self._radius, bias=self._bias)

def callback(t: int, count: int, channels, tau, radius, bias) -> Callback:
    def seq(a, b, c):
        """Returns one of len(c) responses from <c>.  The equation divides <b> elements evenly into <len(c)> partitions.
        The equation then uses integer division returning the partition of the <a>th offset into <b> elements"""
        return c[int(np.floor(a / (b / len(c))))]

    f: Callback = partial(gen,
                          tau=tau,
                          func=[partial(sine,
                                        phase=xx,
                                        radius=seq(t, count, radius),
                                        bias=seq(t, count, bias)) for xx in
                                np.linspace(0, 1, channels, endpoint=True)])
    return f


def show(filename: str, data: List[np.ndarray]) -> None:
    """
    Create a plot for the provided data.  It is assumed that the data (k x m x n) divided into multiple panels (dim 0),
    with the remaining two dimensions (m & n) making up the body of the graph.  It is assumed that m is the number
    of individual feeds and n is the data of those feeds.
    :param filename:
    :param data:
    :return:
    """
    print("show(filename={}, data=len({}))".format(filename, len(data)))

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


def evaluate(neuron: Neuron, f_in3, filename: str) -> None:
    print("evaluate({}, {})".format(neuron, filename))

    # DEPTH OF EVALUATION DATA - SMALLER = LOWER RESOLUTION
    cnt = len(f_in3)

    # Setup local variables to hold all return values
    u = np.zeros((cnt, neuron.numInputs))
    x = np.zeros((cnt, neuron.numReservoir))
    y = np.zeros((cnt, neuron.numOutputs))
    z = np.zeros((cnt, neuron.numReservoir + neuron.numInputs))

    # Generic input patterns to try
    f_in1 = [callback(n, cnt, neuron.numInputs, cnt / 2, radius=[0], bias=[0.0]) for n in range(cnt)]
    f_in2 = [callback(n, cnt, neuron.numInputs, cnt / 2, radius=[0], bias=[0.5]) for n in range(cnt)]

    # Run Network
    for idx, f_in in enumerate([f_in1, f_in2, f_in3]):
        for n in range(cnt):
            zzz = neuron.cycle(n, n, [f_in[n]])
            u[n] = zzz[0].reshape(zzz[0].shape[0])
            x[n] = zzz[1].reshape(zzz[1].shape[0])
            y[n] = zzz[2].reshape(zzz[2].shape[0])
            z[n] = zzz[3].reshape(zzz[3].shape[0])
        show(filename.format(idx), [u, x, y])


def train(neuron: Neuron, f_in: Sequence[Callback], f_out: Sequence[Callback], filename: str) -> None:
    print("train(neuron={})".format(neuron))

    # DEPTH OF TRAINING DATA - SMALLER = LOWER RESOLUTION
    cnt = len(f_in)

    # # Setup local variables to hold all return values
    # u = np.zeros((cnt, neuron.numInputs))
    # x = np.zeros((cnt, neuron.numReservoir))
    # y = np.zeros((cnt, neuron.numOutputs))
    # z = np.zeros((cnt, neuron.numReservoir + neuron.numInputs))
    #
    # # VISUALIZE RESPONSE PRIOR TO TRAINING
    # for n in range(cnt):
    #     zzz = neuron.cycle(n, n, [f_in[n]])
    #     u[n] = zzz[0].reshape(zzz[0].shape[0])
    #     x[n] = zzz[1].reshape(zzz[1].shape[0])
    #     y[n] = zzz[2].reshape(zzz[2].shape[0])
    #     z[n] = zzz[3].reshape(zzz[3].shape[0])
    # show("esn0_VisualizeTrainingInput0.png", np.arange(cnt), [u, x, y])

    # TRAINING BLOCK ####################################################
    S, D = neuron.learn(0, cnt - 1, f_in, f_out)
    uu = S[:, -neuron.numInputs:]
    xx = S[:, 0: neuron.numReservoir]
    show(filename, [uu, xx, D])
    # TRAINING BLOCK ####################################################

    # # VISUALIZE RESPONSE SUBSEQUENT TO TRAINING
    # for n in range(cnt):
    #     zzz = neuron.cycle(n, n, [f_in[n]])
    #     u[n] = zzz[0].reshape(zzz[0].shape[0])
    #     x[n] = zzz[1].reshape(zzz[1].shape[0])
    #     y[n] = zzz[2].reshape(zzz[2].shape[0])
    #     z[n] = zzz[3].reshape(zzz[3].shape[0])
    # show("esn0_VisualizeTrainingInput2.png", np.arange(cnt), [u, x, y])


def load_model(loadExisting: bool = False) -> Neuron:
    print("load_model(loadExisting={})".format(loadExisting))

    neuron = Neuron.load() if loadExisting else None
    if neuron is None:
        neuron = Neuron(
            numInputs=1,
            numReservoir=1000,
            numOutputs=1,
            pct=.50,
            alpha=0.9,
            f=Neuron.tanh,
            g=Neuron.tanh,
            feedback=False)
        neuron.save()
    return neuron


def main() -> int:
    print("main()")
    try:
        neuron = load_model(False)
        neuron.settle(1000)

        cnt = 400

        # f_in=[callback(n, cnt, neuron.numInputs, radius=[0, 1], bias=[0.0, 0.5, 0.0, -0.5, 0.0]) for n in range(cnt)]
        f_in = [callback(n, cnt, neuron.numInputs, cnt / 10, radius=[1, 0], bias=[0.0]) for n in range(cnt)]
        f_out = [callback(n, cnt, neuron.numOutputs, cnt / 10, radius=[0, 0.9], bias=[0.1]) for n in range(cnt)]

        # Show initial capability
        evaluate(neuron, f_in, "esn0_0Pre_{}.png")

        # Learn to treat 0 input and 0.5 input as the same output
        target = [callback(n, cnt, neuron.numOutputs, cnt / 10, radius=[0], bias=[0.0]) for n in range(cnt)]
        for bias in [0.0, 0.5]:
            source = [callback(n, cnt, neuron.numInputs, cnt / 2, radius=[0], bias=[bias]) for n in range(cnt)]
            train(neuron, source, target, "esn0_TrainingData_bias_{}.png".format(bias))

        # Show what we've learned
        evaluate(neuron, f_in, "esn0_1Flat_{}.png")

        # Learn the input to output pattern we want
        train(neuron, f_in, f_out, "esn0_TrainingData_Final.png")

        # Lets wait around a while with "null" input
        neuron.settle(1000)

        evaluate(neuron, f_in, "esn0_2Post_{}.png")
    except Exception as erc:
        print("Exception Encountered:", erc)
        sys.stdout.flush()
        sys.stderr.flush()
        raise erc
    return 0


if __name__ == "__main__":
    np.seterr(all="raise")
    np.set_printoptions(precision=4, linewidth=512, floatmode='fixed', suppress=True, sign=' ')
    sys.exit(main())
