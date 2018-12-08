import pickle

import numpy as np


class Neuron:
    """
    In the framework of the Echo State Neuron, an individual neuron is vastly more complicated and capable than
    traditional neural network nodes.  In the traditional case the dot product of an input vector and a learned
    weight vector plus a bias are passed through some non linearity (or not) to produce the neuron's activation.  The
    formulation is thus:

    $$ a_{out} = \sigma(\vec{a_{in}} \cdot \vec{w} + b) $$

    The basic problem with this is that it is static. It completely misses the dynamics possible in a real neuron and
    does now allow for spatiotemporal receptive field mapping without playing with approximations of time,
    etc. And even then, the output really does not vary based on the input to the system.

    I propose an ESNeuron.  The ESNeuron gets its name from echo state networks.  In an ESNeuron multiple inputs
    still exist and there is a single output.  The distinction comes when everything else is considered.  In an
    ESNeuron the inputs are fully connected to a dynamical pool.  If we assume learning or processing occurs in the
    dendritic arbor of a neuron, then the dynamical pool is an analog to the dendritic arbor.  We can likewise assume
    that the axon hillock is the integration point in a neuron. In this case, our ESNeuron performs a linear
    transformation with the reservoir resulting in a value taken to be the ESNeuron activation.

    The final step in the system is to connect the output via fixed weights to the reservoir.  Because we're passing
    through an echo state network, our output is not entirely driven by the input to the system but works with inputs
    received, feedback from prior outputs and the internal dynamical nature of the ESN.

    In theory any output pattern can be trained given any input.  One could produce, given some random fixed pattern
    virtually any wave form.  This wave form can act as the input to another ESN.  In that case, the ESN receives an
    input pattern over time and this drives another potentially more complex output pattern.

    Given the above there are several parameters that decide the ESNeuron:
    """

    __version__ = "0.9"

    @classmethod
    def load(cls):
        try:
            with open("esn_parameters.pkl", "rb") as inp:
                x = pickle.load(inp)
            assert isinstance(x, cls), "Pickled object is not a {}".format(cls)
            return x
        except FileNotFoundError:
            return None

    def __init__(self,
                 numInputs: int = 0,
                 numOutputs: int = 1,
                 numReservoir: int = 0,
                 *,
                 pct: float = 1.0,
                 alpha: float = 1.0,
                 f: callable = None,
                 g: callable = None,
                 feedback: bool = True
                 ) -> None:
        """
        Initialize the neuron with the architecture defined by the parameters

        :param numInputs: The number of connections from precursor neurons to this neuron
        :param numOutputs: The number of connections to the internal integrator from the reservoir
        :param numReservoir: The number of nodes in the internal echo state network
        :param pct: The probability of a connection between nodes of the echo state network
        :param alpha: float: measure of speed of reservoir lower values faster 0 < alpha <= 1
        :param f: callable: default Neuron.tanh
        :param g: callable: default Neuron.identity
        :param feedback: bool: default True - feed outputs back into the reservoir
        """

        assert 0 < pct <= 1, "Probability of connection must be between 0 exclusive and 1 inclusive"
        assert numInputs >= 0, "We restrict the number of inputs to be non-negative"
        assert numOutputs > 0, "We restrict the number of outputs from the reservoir to be strictly positive"

        self.alpha = alpha
        self.f = f if f is not None else Neuron.tanh
        self.g = g if g is not None else Neuron.identity
        self.pct = pct
        self.feedback = feedback

        self.K = numInputs
        self.N = numReservoir
        self.L = numOutputs

        self.default_stimulus = np.zeros(self.K)
        self.default_response = np.zeros(self.L)

        # The number of interations to retain in u, y and u.
        self.history = 1000

        self.u = np.zeros((self.history, self.K))
        self.x = np.zeros((self.history, self.N))
        self.y = np.zeros((self.history, self.L))
        self.d = np.zeros((self.history, self.L))

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
        self.W_out = np.random.randn(self.L, self.K + self.N)

        # Set up the matrix holding the precursor values of the output layer
        self.z = np.zeros((self.history, self.K + self.N))

        # Set up the variable used to count number of iterations
        self.n = 0  # used to reference states

    @staticmethod
    def logistic(z: np.ndarray) -> np.ndarray:
        return 1 / (2 + np.expm1(-z))

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def softplus(z: np.ndarray) -> np.ndarray:
        return np.log(np.expm1(z) + 2)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.array([max(0, zz) for zz in z])

    @staticmethod
    def relu6(z: np.ndarray) -> np.ndarray:
        return np.array([min(max(zz, 0), 6) for zz in z])

    @staticmethod
    def binary(z: np.ndarray) -> np.ndarray:
        return np.array([[-1, 1][zz > 0] for zz in z])

    @staticmethod
    def identity(z: np.ndarray) -> np.ndarray:
        return z

    def save(self) -> None:
        with open("esn_parameters.pkl", "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    def _get_vars(self, varname: np.ndarray) -> np.ndarray:
        if self.n < self.history:
            return varname[0:self.n]
        else:
            return varname[[(self.n + i) % self.history for i in range(self.history)]]

    def get_u(self) -> np.ndarray:
        return self._get_vars(self.u)

    def get_x(self) -> np.ndarray:
        return self._get_vars(self.x)

    def get_y(self) -> np.ndarray:
        return self._get_vars(self.y)

    def _step(self) -> (int, int):
        cur = self.n % self.history
        nxt = (cur + 1) % self.history
        self.n += 1
        return cur, nxt

    def cycle(self, stimulus: np.ndarray = None) -> np.ndarray:
        """
        Execute a single pass through the echo state network potentially taking in some stimulus.
        There is no learning when using this method. To train your ESN, please use <code>force_learn</code>.
        :param stimulus: np.ndarray: a tensor containing an input the ESN.
        :return: the current output vector
        """
        # We keep track of each step we take for <history> steps - wrapping around in that buffer as necessary
        cur, nxt = self._step()
        stimulus = self.default_stimulus if stimulus is None else stimulus

        self.u[nxt] = stimulus

        self.x[nxt] = self.f(
            np.matmul(self.W, self.x[cur]) +
            np.matmul(self.W_in, self.u[nxt]) +
            np.matmul(self.W_back, self.y[cur]))

        self.z[cur] = np.concatenate((self.x[cur], self.u[cur]))
        self.y[cur] = self.g(np.matmul(self.W_out, self.z[cur]))

        return self.y[cur]

    def force_learn(self, stimulus: np.ndarray = None, response: np.ndarray = None):
        """
        (1) x(n+1) = f(W * x(n) + Win * u(n+1) + Wfb * y(n))

        z(n) = [x(n); u(n)]

        (2) y(n) = g(Wout * z(n))

        Learning equations. In the state harvesting stage of the training, the ESN is driven by an input sequence
        u(1),…,u(nmax) , which yields a sequence z(1),…,z(nmax) of extended system states. The system equations (1),
        (2) are used here. If the model includes output feedback (i.e., nonzero Wfb), then during the generation of the
        system states, the correct outputs d(n) (part of the training data) are written into the output units ("teacher
        forcing"). The obtained extended system states are filed row-wise into a state collection matrix S of size
        nmax×(N+K) . Usually some initial portion of the states thus collected are discarded to accommodate for a
        washout of the arbitrary (random or zero) initial reservoir state needed at time 1. Likewise, the desired
        outputs d(n) are sorted row-wise into a teacher output collection matrix D of size nmax×L .

        The desired output weights Wout are the linear regression weights of the desired outputs d(n) on the harvested
        extended states z(n) . A mathematically straightforward way to compute Wout is to invoke the pseudoinverse
        (denoted by ⋅†) of S :

        (3) Wout = (S†D)′

        which is an offline algorithm (the prime denotes matrix transpose). Online adaptive methods known from linear
        signal processing can also be used to compute output weights (Jaeger 2003).

        Execute a single forced learning phase taking stimulus and desired response into consideration.
        :param stimulus: np.ndarray: an input structure fed
        :param response: np.ndarray: the desired response for the given input
        :return:
        """
        # We keep track of each step we take for <history> steps - wrapping around in that buffer as necessary
        cur, nxt = self._step()
        stimulus = self.default_stimulus if stimulus is None else stimulus
        response = self.default_response if response is None else response

        self.u[nxt] = stimulus
        self.d[nxt] = response

        self.x[nxt] = self.f(
            np.matmul(self.W, self.x[cur]) +
            np.matmul(self.W_in, self.u[nxt]) +
            np.matmul(self.W_back, self.y[cur]))

        self.z[cur] = np.concatenate((self.x[cur], self.u[cur]))
        self.y[cur] = self.g(np.matmul(self.W_out, self.z[cur]))

        return self.y[cur]
