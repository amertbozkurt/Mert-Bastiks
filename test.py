print("hello world")

import numpy as np


def circumference(r):
    """ This function calculates
    the circumference of a circle
    with radius r"""
    return 2.0*np.pi*r


def surface_area(r):
    '''
	This function returns the surface area of a circle.
    ''' 
    return np.pi*r**2

def disk(x,y,r):
    return x**2 + y**2 <= r**2