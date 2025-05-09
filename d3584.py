# Copyright © 2024 Giovanni Squillero <giovanni.squillero@polito.it>
# https://github.com/squillero/computational-intelligence
# Free under certain conditions — see the license for details.

import numpy as np

# All numpy's mathematical functions can be used in formulas
# see: https://numpy.org/doc/stable/reference/routines.math.html


# Notez bien: No need to include f0 -- it's just an example!
def f0(x: np.ndarray) -> np.ndarray:
    return np.tan(x[0] - x[1] + np.tan(x[0])) + 0.15322121158548999 * np.sin(x[1]) / x[1]


def f1(x: np.ndarray) -> np.ndarray: 
    return 0.3782860327383275 * x[0] / (x[0] - 0.7458863362755191)


def f2(x: np.ndarray) -> np.ndarray:
    term1 = (-1112267.7584117618 * x[0] - 1112267.7584117618 * np.sin(1.9842202415071949 * np.sin(x[0])))
    term2 = np.sqrt(-np.sqrt(x[1] + x[2]) * np.sin(np.sin(x[0])) + np.sin(x[2]))
    term3 = np.sqrt(x[1] - x[2] / np.sin(np.sin(x[0])) - np.tan(np.sin(x[0])))
    term4 = (x[0] + np.sin(1.50996183095054 * np.sin(x[0])))
    term5 = (-2189.267872464636 * x[2] * (x[1] + x[2] - np.tan(np.sin(x[0]))) + 2742717.0591339385)
    
    return term1 * term2 * term3 + term4 * term5


def f3(x: np.ndarray) -> np.ndarray:
    return (2 * x[0]**2 + 1.888011443473933 * x[0] - 5.751754953796507 * x[1]**2 + 
            x[1] * (-x[0] * x[1] - np.tan(np.tan(x[2])) + 7.960067489897881) - 
            1.866842958998818 * x[1] - 3.499972972522648 * x[2] + 3.343913409679408)


def f4(x: np.ndarray) -> np.ndarray:
    term1 = np.sqrt(0.59442506135513196 * x[0] - 10.843735353774745)
    term2 = (0.01074365698385143 * np.exp(0.50406178400871155 * x[0]**3 - 
                                        np.sin(2 * np.cos(x[1]))) + 
             6.990332232812374) * np.cos(x[1])
    return term1 + term2


def f5(x: np.ndarray) -> np.ndarray:
    complex_term = (374.93213046082964 * x[0]**2 * x[1]**3 + 
               (-x[0] - x[1] + np.exp(x[0])) * np.exp(2 * x[1]) - 
               0.2477463143761617 * np.exp(0.4441536940414833 * x[0] + x[1]) - 
               34390.847090009492 * np.exp(x[0] + np.sin(np.sin(x[0]))) - 
               np.exp(np.exp(x[1])) - 24175.980509968689)
    
    return -6.226751041691537e-16 * x[0] * x[1] * complex_term


def f6(x: np.ndarray) -> np.ndarray:
    sqrt_term = np.sqrt(np.sqrt(np.sqrt(-1.8923799511965012 - 1.6226422471794368) / x[1]))
    return np.tan(np.sqrt(-1.0268830067559338)) * (0.26623240720729413 * sqrt_term)


def f7(x: np.ndarray) -> np.ndarray: 
    return 9.431522071392338 * np.sqrt(np.sqrt(x[0]) * (x[1] + 1.605583320053832) * np.log(-x[0]))

def f8(x: np.ndarray) -> np.ndarray: 
    term1 = -x[1] - x[4] + np.sqrt(x[5]) - 3 * x[5] - 39.027296247012435 * np.sqrt(-x[4])
    term2 = (-x[4] + 39.027296247012435 * np.sqrt(-x[5]) + 1523.1298523520709) * np.exp(-x[0])
    term3 = -5 * np.exp(x[4]) + np.exp(2 * x[5]) - 2 * np.exp(x[5]) + 11691.94197593936
    
    return term1 - term2 + term3