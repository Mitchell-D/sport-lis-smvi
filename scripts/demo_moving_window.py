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

## proving the first element in the input array is ignored...
assert np.isclose(ma1[0], np.average(x[1:window+1]))

## method 2: final size is size - window + 1
## each element is mean of previous window-1 elements and itself, so there are
## no members of the input array that are not explicitly used in an average.
flt = np.ones(window) / window
ma2 = np.apply_along_axis(
        lambda m:np.convolve(m,flt,mode="valid"),
        axis=0,
        arr=x
        )

## proving method 2 doesn't ignore any elements  from the input array
assert np.isclose(ma2[0], np.average(x[:window]))
assert np.isclose(ma2[-1], np.average(x[-window:]))

## method 3, showing that one fewer element is needed for the same result.
ma3 = np.apply_along_axis(
        lambda m:np.convolve(m,flt,mode="valid"),
        axis=0,
        arr=x[1:]
        )

print(f"ma1:{ma1.shape} ma2:{ma2.shape} ma3:{ma3.shape}")

assert np.all(np.isclose(ma1, ma2[1:]))
assert np.all(np.isclose(ma1, ma3))
