from numpy import random, zeros, transpose, savetxt
from math import sqrt, pi
from numba import njit, types, prange
from numba.experimental import jitclass
import time


random.seed(9)

GAMMA = sqrt(pi)


@njit(fastmath=True, cache=True)
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
    There are two method:

    1.   getIntegral: calculate particular integration
    2.   getDataset: return a set of integration value for plotting and NN training
    """
    def __init__(self, a: int, b: int, N: int, dx: float, data_count: int):
        self.a = a
        self.b = b
        self.N = N
        self.dx = dx
        self.n = data_count
    
    @staticmethod
    def f(t: float, x: float):
        temp = t / (sqrt(x - t) * GAMMA)
        return temp

    def getIntegral(self):
        # O(self.N)
        ar = zeros(self.N)
        for i in prange(self.N):
            ar[i] = random.uniform(self.a, self.b)
        integral = 0.0
        for i in ar:
            integral += self.f(i, self.b)
        ans = (self.b - self.a) / float(self.N) * integral
        return ans
    
    def getDataset(self):
        # O(self.n * self.N)
        store = zeros(self.n)
        index = zeros(self.n)
        for y in prange(1, self.n+1):
            b = self.a + y * self.dx
            ar = zeros(self.N)
            for i in prange(self.N):
                ar[i] = random.uniform(self.a, b)
            integral = 0.0 
            for i in ar:
                integral += self.f(i, b)
            store[y-1] = (b - self.a) / float(self.N) * integral
            index[y-1] = b
        return index, store


def main():
    integration = MCIntegration(a=0, b=1, N=1000000, dx=0.001, data_count=1000)

    ans = integration.getIntegral()
    expected = correctFunction(0.5)
    print("The error is:", abs(expected - ans))
    print(ans)

    index, store = integration.getDataset()
    correct = correctFunction(index)

    # Export caculated dataSet and correct dataSet to CSV File
    name = "1"
    export = transpose([index, store])
    savetxt("function" + name + ".csv", export, delimiter=", ")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Finished in {time.perf_counter() - start :f} second(s)!")
