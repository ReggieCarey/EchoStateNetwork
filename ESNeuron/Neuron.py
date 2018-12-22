import pickle
from typing import Tuple, Callable, Sequence

import numpy as np

__all__ = ["Neuron", "Callback"]

Callback = Callable[[float], np.ndarray]


class BaseNeuron:
    @staticmethod
    def logistic(z: np.ndarray) -> np.ndarray:
        return 1 / (2 + np.expm1(-z))

    @staticmethod
    def tanh(z: np.ndarray, *args, **kwargs) -> np.ndarray:
        return np.tanh(z, *args, **kwargs)

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


class Neuron(BaseNeuron):
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

    def __init__(self,
                 numInputs: int,
                 numReservoir: int,
                 numOutputs: int,
                 pct: float,
                 alpha: float,
                 *,
                 f: Callable[[np.ndarray], np.ndarray] = BaseNeuron.tanh,
                 g: Callable[[np.ndarray], np.ndarray] = BaseNeuron.identity,
                 feedback: bool = True
                 ):
        """
        Initialize the neuron with the architecture defined by the parameters

        :param numInputs: The number of connections from precursor neurons to this neuron
        :param numReservoir: The number of nodes in the internal echo state network
        :param numOutputs: The number of connections to the internal integrator from the reservoir
        :param pct: The probability of a connection between nodes of the echo state network
        :param alpha: float: measure of speed of reservoir lower values faster 0 < alpha <= 1
        :param f: callable: default Neuron.tanh
        :param g: callable: default Neuron.identity
        :param feedback: bool: default True - feed outputs back into the reservoir
        """
        assert 0 <= numInputs, "We restrict the number of inputs to be non-negative"
        assert 0 < numReservoir, "We restrict the number of reservoir nodes to be strictly positive"
        assert 0 < numOutputs, "We restrict the number of outputs to be strictly positive"

        assert 0 < pct <= 1, "Probability of connection must be between 0 exclusive and 1 inclusive"
        # assert 0 < alpha < 1, "alpha - must be 0 < alpha < 1"

        assert callable(f), "f must be a function(arg) or None"
        assert callable(g), "f must be a function(arg) or None"

        # These parameters are sufficient to describe the echo state network dimensions
        self._numInputs = numInputs  # I  K
        self._numReservoir = numReservoir  # H  N
        self._numOutputs = numOutputs  # O  L

        # Store the functions used to define how the neuron operates
        self.f = f
        self.g = g

        # Construct the Input Matrix that transforms the input vector from length I to length H
        self.Win = np.random.standard_normal((numReservoir, numInputs))

        # Construct the Reservoir Matrix that transforms Hidden to Hidden.
        self.W = np.eye(numReservoir)
        self.W[np.random.rand(numReservoir, numReservoir) < pct] = 1
        self.W[self.W > 0] = np.random.standard_normal((numReservoir, numReservoir))[self.W > 0]
        self.W = self.W * alpha / np.max(np.abs(np.linalg.eig(self.W)[0]))

        # Construct the feedback network
        self.Wfb = np.random.randn(numReservoir, numOutputs) * int(feedback)

        # Construct the output network
        self._Wout = np.random.standard_normal((numOutputs, numReservoir + numInputs))
        self.baseWout = self._Wout.copy()
        # self._Wout = np.zeros(shape=(numOutputs, numReservoir + numInputs))

        # Construct the basic activation vectors (neurons)
        self.x = np.random.uniform(-1, 1, size=(numReservoir, 1))
        self.y = np.random.uniform(-1, 1, size=(numOutputs, 1))
        self.z = np.random.uniform(-1, 1, size=(numReservoir + numInputs, 1))

        # number of iterations of no input (zero vector) used internally as setup and during training
        self._washout = 500

        # make sure the network has settled prior to finishing initialization
        self.forget()
        # self.settle(self.washout)

    @property
    def washout(self):
        return self._washout

    @washout.setter
    def washout(self, washout):
        self._washout = washout

    @property
    def Wout(self) -> np.ndarray:
        return self._Wout

    @Wout.setter
    def Wout(self, Wout: np.ndarray):
        assert self._Wout.shape == Wout.shape, "Cannot set weights. The dimensions do not match"
        self._Wout = Wout.copy()

    @property
    def numInputs(self):
        return self._numInputs

    @property
    def numReservoir(self):
        return self._numReservoir

    @property
    def numOutputs(self):
        return self._numOutputs

    @classmethod
    def load(cls):
        try:
            with open("esn_parameters.pkl", "rb") as inp:
                x = pickle.load(inp)
            assert isinstance(x, cls), "Pickled object is not a {}".format(cls)
            assert x.__version__ == cls.__version__, "Version mismatch - pkl file is out of date"
            return x
        except FileNotFoundError:
            return None

    def save(self):
        with open("esn_parameters.pkl", "wb") as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    def forget(self):
        """
        Forget any prior learning by zeroing the output weight matrix and then running the network with
        no input for a <washout> number of steps.
        :return: None
        """

        # Zero the output matrix and then let the network settle for the washout period
        # self._Wout = np.zeros(shape=self._Wout.shape)
        self._Wout = self.baseWout.copy()
        self.settle(self.washout)

    def settle(self, count: int):
        """
        Cause the network to run <count> steps with no input.  This is intended to remove prior input traces
        from the reservoir prior to training or when ever a clean slate is desired. The number of steps <count>
        really should be computed based on the value of alpha - the spectral radius of the Reservoir.
        :param count: int; The number of cycles to run the ESN with no input.
        :return: None
        """

        def fin(_: float) -> np.ndarray:
            """function returning zero vector size numInputs"""
            return np.zeros((self.numInputs, 1))

        self.cycle(1, count, [fin for _ in range(count)])

    def cycle(self,
              t0: int,
              tn: int,
              f_in: Sequence[Callback]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        (1) x(n+1) = f(W * x(n) + Win * u(n+1) + Wfb * y(n))
            z(n) = [x(n); u(n)]
        (2) y(n) = g(Wout * z(n))

        Execute a single pass through the echo state network potentially taking in some stimulus.
        There is no learning when using this method. To train your ESN, please use <code>force_learn</code>.
        :param t0: int: Starting index passed to f_teach to retrieve the t0'th teaching pair
        :param tn: int: Ending index passed to f_teach to retrieve the tn'th teaching pair
        :param f_in: callable(t: float) -> np.array a function that returns the network input at time t
        :return: the current output vector
        """
        assert t0 <= tn, "Cannot observe backwards through time. tn must not be less than t0"

        nmax = tn - t0 + 1
        u = None

        for t in range(nmax):
            u = f_in[t](t0 + t)  # retrieve the next input pattern
            self.x = self.f((self.W @ self.x) + (self.Win @ u) + (self.Wfb @ self.y))
            self.z = np.concatenate((self.x, u))
            self.y = self.g(self._Wout @ self.z)

        return u, self.x, self.y, self.z

    def learn(self,
              t0: int,
              tn: int,
              f_in: Sequence[Callback],
              f_out: Sequence[Callback]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute a single forced learning phase taking stimulus and desired response into consideration.

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
             S = z(t0:tn)
             D = y(t0:tn)
             Wout = np.matmul(np.linalg.pinv(S), D).T

        which is an offline algorithm (the prime denotes matrix transpose). Online adaptive methods known from linear
        signal processing can also be used to compute output weights (Jaeger 2003).

        A note on the function f_teach.  f_teach(t) should return a tuple of functions where the first
        of the tuple, when called returns the t'th input pattern and the second tuple, when called returns
        the t'th target pattern.  These returned functions take no parameters and each returns a vector.

        :param t0: int: Starting index passed to f_teach to retrieve the t0'th teaching pair
        :param tn: int: Ending index passed to f_teach to retrieve the tn'th teaching pair
        :param f_in: callable(t: int) -> np.array a function that returns the network input at time t
        :param f_out: callable(t: int) -> np.array a function that returns the target output at time t
        :return:
        """
        assert t0 <= tn, "Cannot train backwards through time. tn must not be less than t0"

        nmax = tn - t0 + 1

        S = np.empty(shape=(nmax, self.numReservoir + self.numInputs))
        D = np.empty(shape=(nmax, self.numOutputs))

        print("Learning sequence of {} inputs".format(nmax))
        for t in range(nmax):
            # self.cycle(t0 + t, t0 + t, f_in)

            u = f_in[t](t0 + t)  # retrieve the next input pattern
            y = f_out[t](t0 + t)  # retrieve the next output pattern

            reservoir = (self.W @ self.x)
            inputData = (self.Win @ u)
            feedBack = (self.Wfb @ self.y)

            self.x = self.f(reservoir + inputData + feedBack)
            self.z = np.concatenate((self.x, u))
            self.y = self.g(self._Wout @ self.z)

            # Teacher Forcing
            self.y = y

            # Save our state and target outputs
            S[t] = self.z.copy().reshape(-1)
            D[t] = y.copy().reshape(-1)

        self._Wout += (np.linalg.pinv(S) @ D).T
        return S, D
