from Neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt


def main():
    n = None
    # n = Neuron.load()
    if n is None:
        n = Neuron(numInputs=2, numOutputs=10, numReservoir=30, pct=1, feedback=True, f=Neuron.tanh, g=Neuron.tanh)
        n.save()

    cnt = 200

    inp = np.ones(n.K)

    for i in range(cnt):
        if i < cnt * 0 // 8 or i > cnt * 4 // 8:
            n.cycle()
        else:
            inp[0] = np.sin(np.pi * i / 20)
            inp[1] = np.cos(np.pi * i / 10)
            n.cycle(inp)

    fig, ax = plt.subplots(1, 3, sharex="all", figsize=(15, 5))

    data = n.get_u().copy()
    for i in range(n.K):
        data[:, i] = data[:, i] + (2.5 * i)

    i = 0
    # [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
    ax[i].patch.set_facecolor((.3, .6, .9, .1))
    ax[i].plot(data, lw=1, color="red")
    ax[i].set_ylabel("$u$")
    # ax[i].set_ylim((-1.1, 1.1))
    ax[i].set_yticks([])

    data = n.get_x().copy()
    for i in range(n.N):
        data[:, i] = data[:, i] + (2.5 * i)

    i = 1
    [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
    ax[i].patch.set_facecolor((.3, .6, .9, .1))
    ax[i].plot(data, lw=0.5)
    ax[i].set_ylabel("$x$")
    # ax[i].set_ylim((-1.1, 1.1))
    ax[i].set_yticks([])

    data = n.get_y().copy()
    for i in range(n.L):
        data[:, i] = data[:, i] + (2.5 * i)

    i = 2
    [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
    ax[i].patch.set_facecolor((.3, .6, .9, .1))
    ax[i].plot(data, lw=1, alpha=1, color="green")
    ax[i].set_ylabel("$y$")
    # ax[i].set_ylim((-1.1, 1.1))
    ax[i].set_yticks([])

    print("Graphing")
    fig.show()
    fig.savefig("esn.png")
    plt.close(fig)


if __name__ == "__main__":
    np.seterr(all="raise")
    print("Starting")
    main()
    print("Done")
