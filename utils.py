import cv2
import numpy
import scipy.interpolate

def createCurveFunc(points):
    '''return a function derived from control points.'''
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None
    xs, ys = zip(*points)
    if numPoints < 3:
        kind = 'linear'
    elif numPoints < 4:
        kind = 'quadratic'
    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs, ys, kind, bounds_error=False)


def createLookupArray(func, length=256):
    '''return a lookup for whole number inputs to a function'''
    if func is None:
        return None
    lookupArray = numpy.empty(length)
    i = 0
    while i < length:
        func_i = func(i)
        lookupArray[i] = min(max(0, func_i), length-1)
        i += 1
    return lookupArray

def applyLookupArray(lookupArray, src, dst):
    '''map a source to a destination using a lookup'''
    if lookupArray is None:
        return
    dst[:] = lookupArray[src]

def createCompositeFunc(func0, func1):
    '''return a composite of two functions'''
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))


