import matplotlib.pyplot as plt
import numpy as np
import pickle


def logistic(z):
    return 1 / (2 + np.expm1(-z))


def tanh(z):
    return np.tanh(z)


def softplus(z):
    return np.log(np.expm1(z) + 2)


def relu(z):
    return [max(0, zz) for zz in z]


def relu6(z):
    return [min(max(zz, 0), 6) for zz in z]


def binary(z):
    return [[-1, 1][zz > 0] for zz in z]


def identity(z):
    return z


class Neuron:
    """
    In the framework of the Echo State Neuron, an individual neuron is vastly more complicated and capable than
    traditional neural network nodes.  In the traditional case the dot product of an input vector and a learned weight
    vector plus a bias are passed through some non linearity (or not) to produce the neuron's activation.  The
    formulation is thus:

    $$ a_{out} = \sigma(\vec{a_{in}} \cdot \vec{w} + b) $$

    The basic problem with this is that it is static. It completely misses the dynamics possible in a real neuron and
    does now allow for spatio temporal receptive field mapping without playing with approximations of time, etc. And
    even then, the output really does not vary based on the input to the system.

    I propose an ESNeuron.  The ESNeuron gets its name from echo state networks.  In an ESNeuron multiple inputs still
    exist and there is a single output.  The distinction comes when everything else is considered.  In an ESNeuron the
    inputs are fully connected to a dynamical pool.  If we assume learning or processing occurs in the dendritic arbor
    of a neuron, then the dynamical pool is an analog to the dendritic arbor.  We can likewise assume that the axon
    hillock is the integration point in a neuron. In this case, our ESNeuron performs a linear transformation with the
    reservior resulting in a value taken to be the ESNeuron activation.

    The final step in the system is to connect the output via fixed weights to the reservoir.  Because we're passing
    through an echo state network, our output is not entirely driven by the input to the system but works with inputs
    received, feedback from prior outputs and the internal dynamical nature of the ESN.

    In theory any output pattern can be trained given any input.  One could produce, given some random fixed pattern
    virtually any wave form.  This wave form can act as the input to another ESN.  In that case, the ESN receives an
    input pattern over time and this drives another potentially more complex output pattern.

    Given the above there are several parameters that decide the ESNeuron:
    """

    def __init__(self,
                 numInputs: int = 0,
                 numOutputs: int = 1,
                 numReservoir: int = 0,
                 *,
                 pct: float = 0.5,
                 alpha: float = 0.1,
                 f: callable = tanh,
                 g: callable = identity,
                 feedback: bool = True
                 ):
        """
        Initialize the neuron with the architecture defined by the parameters

        :param numInputs: The number of connections from precursor neurons to this neuron
        :param numOutputs: The number of connections to the internal integrator from the reservoir
        :param numReservoir: The number of nodes in the internal echo state network
        :param pct: The probability of a connection between nodes of the echo state network
        :param alpha:
        :param f:
        :param g:
        :param feedback:
        """

        # We make sure we can capture any errors in numpy
        np.seterr(all="raise")

        assert 0 < pct <= 1, "Probability of connection must be between 0 exclusive and 1 inclusive"
        assert numInputs >= 0, "We restrict the number of inputs to be non-negative"
        assert numOutputs > 0, "We restrict the number of outputs from the reservoir to be strictly positive"

        self.alpha = alpha
        self.f = f
        self.g = g
        self.pct = pct
        self.feedback = feedback

        self.K = numInputs
        self.N = numReservoir
        self.L = numOutputs

        # The number of interations to retain in u, y and u.
        self.history = 1000

        self.u = np.zeros((self.K, self.history))
        self.x = np.zeros((self.N, self.history))
        self.y = np.zeros((self.L, self.history))

        # Calculate the fixed random weights of the input layer
        self.W_in = np.random.randn(self.N, self.K)

        # Calculate the fixed random weights and connectivity of the reservoir layer
        while True:
            numConnected = int(np.ceil(self.N ** 2 * self.pct))
            connectionOrder = np.random.choice(self.N ** 2, numConnected, False)
            self.W = np.zeros(self.N ** 2)
            self.W[connectionOrder] = np.random.randn(numConnected)
            self.W = self.W.reshape((self.N, self.N))
            eigval, _ = np.linalg.eig(self.W)
            spectral_radius = np.max(np.abs(eigval))
            try:
                self.W = self.W / (self.alpha * spectral_radius)
                break
            except FloatingPointError as erc:
                print("connectionOrder", connectionOrder)
                print("eigval", np.abs(eigval))
                print(erc, ": trying again")

        # Calculate the fixed random weights of the feedback layer
        self.W_back = np.random.randn(self.N, self.L) * int(feedback)

        # Calculate the variable weights of the output layer
        self.W_out = np.random.randn(self.L, (self.K + self.N))

        # Set up the matrix holding the precursor values of the output layer
        self.z = np.zeros((self.L, (self.K + self.N)))

        # Set up the variable used to count number of iterations
        self.n = 0  # used to reference states

    def save(self):
        with open("esn_parameters.pkl", "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load():
        try:
            with open("esn_parameters.pkl", "rb") as inp:
                x = pickle.load(inp)
            return x
        except FileNotFoundError:
            return None

    def get_vars(self, varname):
        if self.n < self.history:
            return varname[:, 0:self.n]
        else:
            return varname[:, [(self.n + i) % self.history for i in range(self.history)]]

    def get_u(self):
        return self.get_vars(self.u)

    def get_x(self):
        return self.get_vars(self.x)

    def get_y(self):
        return self.get_vars(self.y)

    def cycle(self, stimulus=None):
        cur = self.n % self.u.shape[1]
        nxt = (cur + 1) % self.u.shape[1]
        self.n += 1

        self.u[:, nxt] = stimulus if stimulus is not None else np.zeros(self.K)
        self.x[:, nxt] = self.f(
            np.matmul(self.W, self.x[:, cur]) +
            np.matmul(self.W_in, self.u[:, nxt]) +
            np.matmul(self.W_back, self.y[:, cur]))
        self.z = np.concatenate((self.x[:, cur], self.u[:, cur]))
        self.y[:, cur] = self.g(np.matmul(self.W_out, self.z))

        return self.y[:, cur]

    def force_learn(self, stimulus, response):
        """
        Learning equations. In the state harvesting stage of the training, the ESN is driven by an input sequence u(1),…,u(nmax) , which yields a sequence z(1),…,z(nmax) of extended system states. The system equations (1), (2) are used here. If the model includes output feedback (i.e., nonzero Wfb), then during the generation of the system states, the correct outputs d(n) (part of the training data) are written into the output units ("teacher forcing"). The obtained extended system states are filed row-wise into a state collection matrix S of size nmax×(N+K) . Usually some initial portion of the states thus collected are discarded to accommodate for a washout of the arbitrary (random or zero) initial reservoir state needed at time 1. Likewise, the desired outputs d(n) are sorted row-wise into a teacher output collection matrix D of size nmax×L .

The desired output weights Wout are the linear regression weights of the desired outputs d(n) on the harvested extended states z(n) . A mathematically straightforward way to compute Wout is to invoke the pseudoinverse (denoted by ⋅†) of S :


        :param stimulus:
        :param response:
        :return:
        """
        cur = self.n % self.u.shape[1]
        nxt = (cur + 1) % self.u.shape[1]
        self.n += 1

        self.u[:, nxt] = stimulus if stimulus is not None else np.zeros(self.K)
        self.x[:, nxt] = self.f(
            np.matmul(self.W, self.x[:, cur]) +
            np.matmul(self.W_in, self.u[:, nxt]) +
            np.matmul(self.W_back, self.y[:, cur]))
        self.z = np.concatenate((self.x[:, cur], self.u[:, cur]))
        self.y[:, cur] = self.g(np.matmul(self.W_out, self.z))

        return self.y[:, cur]

def test():
    n = Neuron.load()
    if n is None:
        n = Neuron(numInputs=2, numOutputs=2, numReservoir=5, pct=0.5, feedback=True, g=tanh)
        n.save()

    cnt = 200

    inp = np.ones(n.K)

    for i in range(cnt):
        if i < cnt * 1 // 8 or i > cnt * 2 // 4:
            n.cycle()
        else:
            inp[0] = np.sin(np.pi * i / 10)
            inp[1] = np.cos(np.pi * i / 20)
            n.cycle(inp)

    fig, ax = plt.subplots(n.N + n.K + n.L, 1, sharex="all")

    data = n.get_u().T
    start = 0
    for idx in range(n.K):
        i = start + idx
        [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
        ax[i].patch.set_facecolor((.3, .6, .9, .1))
        ax[i].plot(data[:, idx], lw=1, alpha=1)
        ax[i].set_ylabel("$u_{{{}}}$".format(idx))
        ax[i].set_ylim((-1.1, 1.1))
        ax[i].set_yticks([])

    data = n.get_x().T
    start = n.K
    for idx in range(n.N):
        i = start + idx
        [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
        ax[i].patch.set_facecolor((.3, .6, .9, .1))
        ax[i].plot(data[:, idx], lw=0.5)
        ax[i].set_ylabel("$x_{{{}}}$".format(idx))
        ax[i].set_ylim((-1.1, 1.1))
        ax[i].set_yticks([])

    data = n.get_y().T
    start = n.K + n.N
    for idx in range(n.L):
        i = start + idx
        [ax[i].spines[loc].set_color((.3, .6, .9, .5)) for loc in ['top', 'bottom', 'left', 'right']]
        ax[i].patch.set_facecolor((.3, .6, .9, .1))
        ax[i].plot(data[:, idx], lw=1, alpha=1)
        ax[i].set_ylabel("$y_{{{}}}$".format(idx))
        ax[i].set_ylim((-1.1, 1.1))
        ax[i].set_yticks([])

    fig.show()
    fig.savefig("esn.png")
    plt.close(fig)

    # print("min(u)", np.min(n.get_u().T))
    # print("max(u)", np.max(n.get_u().T))
    # print("min(x)", np.min(n.get_x().T))
    # print("max(x)", np.max(n.get_x().T))
    # print("min(y)", np.min(n.get_y().T))
    # print("max(y)", np.max(n.get_y().T))


test()
