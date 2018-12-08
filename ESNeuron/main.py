import matplotlib.pyplot as plt
import numpy as np

from Neuron import Neuron


def main():
    # n = None
    n = Neuron.load()
    if n is None:
        n = Neuron(
            numInputs=4,
            numOutputs=1,
            numReservoir=10,
            pct=0.75,
            alpha=.75,
            f=Neuron.tanh,
            g=Neuron.tanh,
            feedback=True
        )
        n.save()

    # DATA GENERATION
    cnt = 100
    inp = np.ones(n.K)
    start, end = 0.0, 0.2
    freq = 2
    for i in range(cnt):
        if i < (cnt * start) or i > (cnt * end):
            n.cycle()
        else:
            for j in range(n.K):
                phase = np.pi * (j / max(1, n.K - 1))
                if j % 2 == 0:
                    inp[j] = np.sin(phase + (freq * 2 * np.pi * (i / max(1.0, cnt * (end - start)))))
                else:
                    inp[j] = np.cos(phase + (freq * 2 * np.pi * (i / max(1.0, cnt * (end - start)))))
            n.cycle(inp)

    # PREPARING DATA FOR VISUALIZATION
    data = list()
    data.append(n.get_u().copy())
    data.append(n.get_x().copy())
    data.append(n.get_y().copy())
    plot_gap = 3
    for d in data:
        for i in range(d.shape[1]):
            d[:, i] = d[:, i] + (plot_gap * i)

    # VISUALIZATION
    fig, ax = plt.subplots(1, 3, sharex="all", figsize=(15, 5))

    ylabel = ["$u$", "$x$", "$y$"]
    lw = [.5, .5, .5]
    colors = ["red", None, "green"]
    alpha = [1, 1, 1]

    for i in range(3):
        [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
        ax[i].patch.set_facecolor((.3, .6, .9, .1))
        ax[i].set_yticks([])
        ax[i].plot(data[i], lw=lw[i], color=colors[i], alpha=alpha[i])
        ax[i].set_xlabel("steps")
        ax[i].set_ylabel("{} has {} channels".format(ylabel[i], data[i].shape[1]))
        # place horizontal lines in the graph indicating 0 points for each plot line.
        for j in range(data[i].shape[1]):
            ax[i].axhline(y=plot_gap * j, lw=0.5, alpha=0.5)

    print("Graphing")
    # fig.show()
    fig.savefig("esn.png")
    plt.close(fig)


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
