""" Comparison of different approaches for calculating moving averages. """
import numpy as np

size = 1024
window = 5
x = np.random.random_sample(size)

## method 1: final size is size - window
## Since the first element of the result negates the first element of the
## input array, the first input element isn't explicitly used in an average.
cs = np.cumsum(x, axis=0)
ma1 = (cs[window:]-cs[:-window]) / window

## method 2: final size is size - window + 1
## each element is mean of previous window-1 elements and itself, so there are
## no elements in the input array that are not explicitly used in an average.
flt = np.ones(window) / window
ma2 = np.apply_along_axis(
        lambda m:np.convolve(m,flt,mode="valid"),
        axis=0,
        arr=x
        )

## method 3, showing that one fewer element is needed for the same result.
ma3 = np.apply_along_axis(
        lambda m:np.convolve(m,flt,mode="valid"),
        axis=0,
        arr=x[1:]
        )

print(f"{ma1.shape=} {ma2.shape=} {ma3.shape=}")

assert np.all(np.isclose(ma1, ma2[1:]))
assert np.all(np.isclose(ma1, ma3))
