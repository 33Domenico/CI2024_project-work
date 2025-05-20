# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html

def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Protected division: returns a/b or 1 when b is close to zero"""
    return np.divide(a, b, out=np.ones_like(a), where=np.abs(b) > 1e-8)

def safe_log(a: np.ndarray) -> np.ndarray:
    """Protected logarithm: returns log(|a|) or 0 for a close to zero"""
    return np.log(np.abs(a), out=np.zeros_like(a), where=np.abs(a) > 1e-10)

def safe_sqrt(a: np.ndarray) -> np.ndarray:
    """Square root protected: returns sqrt(|a|)"""
    return np.sqrt(np.abs(a))

def safe_exp(a: np.ndarray) -> np.ndarray:
    """Protected exponential: limits input to avoid overflow"""
    return np.exp(np.clip(a, -200, 200))

def safe_sin(a: np.ndarray) -> np.ndarray:
    """Protected sin"""
    return np.sin(np.clip(a, -1000, 1000))

def safe_cos(a: np.ndarray) -> np.ndarray:
    """Protected cos"""
    return np.cos(np.clip(a, -1000, 1000))

def safe_tan(a: np.ndarray) -> np.ndarray:
    """Protected tangent: limits outputs to avoid extreme values"""
    return np.clip(np.tan(a), -200, 200)


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return ((((x[0] + ((-2.8636943484015354 * safe_cos((x[1] - -1.583968308262229))) * safe_exp(-3.2082386888331254))) + (safe_sin(x[1]) * safe_exp(-3.154932922313467))) + (safe_sin((x[1] + (safe_sqrt(safe_tan(3.141592653589793)) * (3.141592653589793 / safe_exp(-3.2082386888331254))))) * safe_exp(-3.154932922313467))) - safe_sqrt(((safe_sqrt(safe_tan(3.141592653589793)) * ((-3.154932922313467 / 0.2607832440731248) / 0.2532699192266387)) * ((x[1] + ((x[1] - -1.0329760905609568) - -1.1123733897283157)) * ((safe_sin(safe_sin(x[1])) - -1.0439752897500634) + safe_cos((-1.9944929251132986 * (x[1] - 3.1410124598253333))))))))


def f1(x: np.ndarray) -> np.ndarray: 
    return safe_sin(x[0])


def f2(x: np.ndarray) -> np.ndarray:
    return (((1430908.1901754038 / safe_exp(safe_sqrt((safe_sqrt((x[2] + x[1])) - 0.0)))) * ((safe_cos((safe_sqrt(x[0]) + 0.0)) * (((x[1] + (x[0] + x[2])) + ((x[2] + x[1]) + x[1])) + ((x[2] + (x[0] + x[1])) + ((x[2] + x[0]) + (x[2] + x[1]))))) + ((x[2] + x[0]) + (x[0] + (safe_sin(x[0]) + (x[0] + (x[0] + x[1]))))))) - (((((((x[0] + x[0]) * x[2]) - safe_log(x[0])) - safe_log(x[0])) - safe_log(safe_sin(safe_exp(x[1])))) * ((((x[1] + (x[1] + x[1])) * (safe_sqrt(-4378885.515735305) + safe_sqrt(-4408770.793543419))) - x[2]) + (((x[0] + x[1]) + ((x[2] + x[1]) / (x[1] + x[1]))) * (safe_sqrt(-4005445.0725207813) + x[2])))) * (safe_sqrt((x[2] + x[1])) - 0.0)))


def f3(x: np.ndarray) -> np.ndarray:
    return ((((((x[2] * safe_sin(1.184037735453649)) - (x[2] - safe_sqrt(safe_cos(1.2158401721189076)))) - safe_log(safe_sqrt(safe_cos(108.78615012548357)))) - (safe_cos(safe_log(((x[2] - 2.674670908359565) - 2.6590687466126757))) - safe_sqrt(((x[2] - safe_sqrt(x[2])) - 3.17864466323484)))) + ((((x[0] * x[0]) - (x[2] - safe_sin(108.78615012548357))) - x[2]) + ((x[0] * x[0]) - x[2]))) - ((x[1] * x[1]) * x[1]))


def f4(x: np.ndarray) -> np.ndarray:
    return (safe_cos((safe_exp((-8.23509920982603 - (safe_cos((x[0] + x[1])) * x[0]))) + ((((x[1] + safe_tan(3.141592653589793)) + safe_tan(3.141592653589793)) + safe_tan(3.141592653589793)) - safe_tan(safe_exp(-10.613659135997567))))) + ((safe_sqrt(((-10.822240096384276 + (x[0] * safe_exp(-0.9405033382101015))) + ((x[0] * safe_cos(-0.9916749730176175)) * safe_exp(-0.9916749730176175)))) + (safe_sin(7.871174350285515) * (safe_sqrt(7.871174350285515) * safe_cos((safe_exp(-7.795847756522948) - x[1]))))) + (safe_sqrt(-10.205488998391495) * safe_cos((((safe_exp(-8.36973460432399) - x[1]) - safe_tan(3.141592653589793)) + safe_tan(3.141592653589793))))))


def f5(x: np.ndarray) -> np.ndarray:
    return ((((x[0] * (x[0] * x[1])) + (x[0] * (safe_log((safe_exp(x[1]) + x[1])) * 4.092327718778701))) * safe_sin((safe_sin(3.141592653589793) * safe_sqrt(safe_log((safe_exp(x[1]) + (3.0580787826247215 + x[1]))))))) * (((safe_exp((x[1] + x[1])) - ((safe_exp(6.5609367685064) - ((5.939230066817374 * x[0]) * safe_exp(x[0]))) / safe_exp(safe_cos((5.939230066817374 + x[1]))))) - (safe_exp(5.083623282487277) * ((((5.547345449854117 * 6.5609367685064) + 4.9787556043961265) - (safe_exp(5.083623282487277) / (x[0] / 5.547345449854117))) / x[1]))) - ((((((5.939230066817374 + 5.939230066817374) * (4.818438684907152 * 5.547345449854117)) * safe_exp(4.9787556043961265)) - (safe_exp((5.083623282487277 + x[1])) - (safe_exp(x[1]) * safe_exp(x[0])))) + (x[0] * ((x[0] * (x[1] * x[0])) * safe_exp((3.0580787826247215 + x[1]))))) / 4.032984299426399)))


def f6(x: np.ndarray) -> np.ndarray:
    return((safe_sqrt(safe_sqrt(8.254213750314564)) * ((x[1] + ((((-10.225547483951923 * 9.025417700203663) * (9.590571516689572 * 8.318079817950606)) * ((x[0] * 8.254213750314564) * safe_tan(3.141592653589793))) - ((x[1] * safe_tan(3.141592653589793)) * ((-9.347525141879945 * 9.590571516689572) * (-11.05365123789452 * 9.590571516689572))))) - ((((-10.225547483951923 * 8.254213750314564) * (8.254213750314564 * 6.944557806606948)) * (((-9.892836400798545 * 8.254213750314564) * (8.254213750314564 * 6.944557806606948)) * ((x[0] * 8.254213750314564) * safe_tan(3.141592653589793)))) - ((((-9.347525141879945 * 8.318079817950606) * safe_tan(3.141592653589793)) * ((-9.36898251373954 * 8.254213750314564) * x[1])) * ((-9.570992156231187 * (8.254213750314564 * 6.944557806606948)) * (-9.36898251373954 * 8.254213750314564)))))) - (((safe_exp(-7.99007532984518) * (2.0257904697066818 * ((x[0] * safe_cos(1.0567228657700947)) + x[1]))) + (x[0] - ((safe_exp((-9.36898251373954 * 2.0257904697066818)) * x[1]) - ((x[1] * safe_tan(3.141592653589793)) * ((-9.347525141879945 * 8.318079817950606) * (-10.225547483951923 * 9.590571516689572)))))) / safe_sqrt(2.074545847057164)))


def f7(x: np.ndarray) -> np.ndarray: 
    return (safe_log((((safe_log(((x[0] * 2.644376733863939) * (x[1] - x[0]))) / (((3.141592653589793 / -19.969108235826695) + x[0]) * (x[1] - x[0]))) / (x[1] * (x[1] - ((3.141592653589793 / -385.6635161890224) + x[0])))) * (safe_log((-20.655152549695234 * (safe_tan(x[1]) * safe_tan(x[0])))) / ((x[0] - x[1]) - ((x[1] * x[0]) * ((x[0] - x[1]) - 0.0)))))) * safe_exp((x[1] * (((x[1] / -253.66421093942046) / (safe_tan((x[0] * 2.644376733863939)) * x[1])) + ((((x[0] * x[1]) * (x[1] * x[0])) / -253.66421093942046) + x[0])))))

def f8(x: np.ndarray) -> np.ndarray: 
    return (((safe_sqrt(10394.330112350359) * ((safe_sin(x[4]) + safe_exp((x[4] - safe_log(x[5])))) / (-1816.4551157931455 * (safe_sqrt(x[5]) + safe_sin(safe_sin(x[5])))))) + (x[5] + ((safe_sin(x[4]) + safe_exp((x[4] - safe_sin(x[5])))) / (-1816.4551157931455 * (((x[5] - x[0]) / x[5]) + x[0]))))) * (((x[5] + safe_log(safe_sin(((3883.249575029571 / 10394.330112350359) * x[5])))) * safe_sqrt((x[5] * ((x[5] * safe_sqrt(12908.835609201817)) * x[5])))) * ((x[5] + safe_log(safe_sin(((3883.249575029571 / -14794.920015342483) * x[5])))) + safe_log(safe_sin(((4053.3369688480975 / -14794.920015342483) * x[5]))))))