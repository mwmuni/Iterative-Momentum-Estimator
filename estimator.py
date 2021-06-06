from matplotlib import pyplot as plt
from pi import str_pi
from functools import lru_cache as cache
from typing import Callable, Tuple
import numpy as np

np.set_printoptions(precision=128)

pi = np.float64(str_pi)

def pi_precision(precision: int) -> float:
    if precision < 0:
        return
    if type(precision) is not int:
        print("Truncating", precision, "to", (new_p := int(precision)))
        precision = new_p
    return int(pi * 10**precision) / 10**precision

@cache
def subtract(t1: tuple, t2: tuple) -> tuple:
    return tuple([e1-e2 for e1,e2 in zip(t1,t2)])

@cache
def multiply(t1: tuple, t2: tuple) -> tuple:
    return tuple([e1*e2 for e1,e2 in zip(t1,t2)])

@cache
def add(t1: tuple, t2: tuple) -> tuple:
    return tuple([e1+e2 for e1,e2 in zip(t1,t2)])

@cache
def divide(t1: tuple, t2: tuple) -> tuple:
    return tuple([e1/e2 for e1,e2 in zip(t1,t2)])

def cumsum(arr: list) -> list:
    for i, v in enumerate(arr):
        if i == 0:
            continue
        arr[i] = arr[i-1] + v
    return arr

def vector_dampening(vel: tuple, damp: Tuple[tuple], iter: int, func: Tuple[Callable]) -> tuple:
    rval = [vel]
    while iter > 0:
        for d, f in zip(damp, func):
            vel = f(vel, d)
        rval.append(vel)
        iter -= 1
    return tuple(rval)


NUM_STEPS = 100

target = pi
ini_damp =  np.float64(.6)

result = 0.
prev_delta = 0.
delta = 1.
delta_diff = 1.

while (result != target) and (delta_diff != 0.):
    ini_vel =  np.float64(1.),
    vel_dampening = (ini_damp,),
    vel_steps = vector_dampening(ini_vel, damp=vel_dampening, iter=NUM_STEPS, func=(multiply,))

    vel_steps = [v[0] for v in vel_steps]
    # print(vel_steps)
    vel_steps = cumsum(vel_steps)
    # print(vel_steps)
    last_val = vel_steps[-1]

    delta = target - last_val
    delta_diff = delta - prev_delta
    print("Initial Velocity:", ini_vel)
    print("Dampening:", ini_damp)
    print("Delta:", delta)
    ini_damp = ini_damp + (delta / 10)
    result = last_val
    print("Pr:", result)
    print("GT:", target)
    prev_delta = delta

# pi_inc_precision = [pi_precision(i) for i in range(16)]
# x_interval = [i/(pi*(i+pi)) for i in range(1, NUM_STEPS+2)]
x_interval = [i for i in range(0, NUM_STEPS+1)]

# plt.plot([i for i in range(len(pi_inc_precision))], pi_inc_precision)
plt.plot(x_interval, vel_steps)
plt.show()