from numpy import random, zeros, transpose, savetxt
from math import sqrt, pi
from numba import njit, types, prange
from numba.experimental import jitclass
import time


random.seed(9)

GAMMA = sqrt(pi)


@njit(fastmath=True, cache=True, nogil=True)
def correctFunction(t: float):
    temp = (4 * t**(3/2))/(3 * GAMMA)
    return temp


spec = [
    ("a", types.int32),
    ("b", types.int32),
    ("N", types.int32),
    ("dx", types.float64),
    ("n", types.int32)
]
@jitclass(spec)
class MCIntegration:
    """
    MCIntegration class take 5 arguments:
    1.   limit a
    2.   limit b
    3.   N uniform distribution
    4.   step deltaX
    5.   number of steps
    """
    def __init__(self, a: int, b: int, N: int, dx: float, data_count: int):
        self.a = a
        self.b = b
        self.N = N
        self.dx = dx
        self.n = data_count

@njit(nogil=True, fastmath=True, cache=True)
def f(t: float, x: float):
        temp = t / (sqrt(x - t) * GAMMA)
        return temp

# calculate particular integration
@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def getIntegral(c: MCIntegration):
    # O(self.N)
    ar = zeros(c.N)
    for i in prange(c.N):
        ar[i] = random.uniform(c.a, c.b)
    integral = 0.0
    for i in ar:
        integral += f(i, c.b)
    ans = (c.b - c.a) / float(c.N) * integral
    return ans

# return a set of integration value for plotting and NN training
@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def getDataset(c: MCIntegration):
    # O(self.n * self.N)
    store = zeros(c.n)
    index = zeros(c.n)
    for y in prange(1, c.n+1):
        b = c.a + y * c.dx
        ar = zeros(c.N)
        for i in prange(c.N):
            ar[i] = random.uniform(c.a, b)
        integral = 0.0
        for i in ar:
            integral += f(i, b)
        store[y-1] = (b - c.a) / float(c.N) * integral
        index[y-1] = b
    return index, store


def main():
    integration = MCIntegration(a=0, b=1, N=1000000, dx=0.001, data_count=1000)

    ans = getIntegral(integration)
    expected = correctFunction(0.5)
    print("The error is:", abs(expected - ans))
    print(ans)

    index, store = getDataset(integration)
    correct = correctFunction(index)

    # Export caculated dataSet and correct dataSet to CSV File
    name = "1"
    export = transpose([index, store])
    savetxt("function" + name + ".csv", export, delimiter=", ")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Finished in {time.perf_counter() - start :f} second(s)!")
