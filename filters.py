import cv2
import numpy
import utils

'''edge detection 
- blur -> to minimize noise detected as an edge features
- grayscale convert
- laplacian operation '''
def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0/255) * (255-graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

class VConvolutionFilter(object):
    '''a filter that applies ta convolution to V'''
    def __init__(self, kernel):
        self._kernel = kernel
    def apply(self, src, dst):
        '''apply the filter with BGR or gray source/destination'''
        cv2.filter2D(src, -1, self._kernel, dst)

class SharpenFilter(object):
    '''a sharpen filter with a 1-pixel radius.'''
    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self, kernel)

class FindEdgesFilter(object):
    '''edge-finding filter with a 1 pixel radius'''
    def __init__(self):
        kernel = numpy.array([[-1,-1,-1],
                              [-1, 8,-1],
                              [-1,-1,-1]])
        VConvolutionFilter.__init__(self, kernel)

class BlurFilter(VConvolutionFilter):
    '''blur filter with a 2 pixel radius'''
    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self, kernel)


'''weight of kernel negative will give sharpen effect, but if weight is positive it will give blur effect'''
class EmbossFilter(object):
    '''an emboss filter with a 1 pixel radius'''
    def __init__(self):
        kernel = numpy.array([[-2,-1,0],
                              [-1, 1,1],
                              [ 0, 1,2]])
        VConvolutionFilter.__init__(self, kernel)


class VFuncFilter(object):
    def __init__(self, vFunc=None, dtype=numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dst):
        '''apply filter with BGR or gray source/destination'''
        srcFlatView = numpy.ravel(src)
        dstFlatView = numpy.ravel(dst)
        utils.applyLookupArray(self._vLookupArray, srcFlatView, dstFlatView)

class VCurveFilter(VFuncFilter):
    def __init__(self, vPoints, dtype=numpy.uint8):
        VFuncFilter.__init__(self, utils.createCurveFunc(vPoints), dtype)

class BGRFuncFilter(object):
    '''a filter that applies different functions to each of BGR.'''
    def __init__(self, vFunc = None, bFunc = None, gFunc = None, rFunc = None, dtype = numpy.uint8):
        length = numpy.iinfo(dtype).max + 1
        self._bLookupArray = utils.createLookupArray(utils.createCompositeFunc(bFunc, vFunc), length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositeFunc(gFunc, vFunc), length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositeFunc(rFunc, vFunc), length)
        
    def apply(self, src, dst):
        b,g,r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray, b, b)
        utils.applyLookupArray(self._gLookupArray, g, g)
        utils.applyLookupArray(self._rLookupArray, r, r)
        cv2.merge([b,g,r], dst)

class BGRCurveFilter(BGRFuncFilter):
    def __init__(self, vPoints=None, bPoints=None, gPoints=None, rPoints=None, dtype=numpy.uint8):
        BGRFuncFilter.__init__(self, 
                               utils.createCurveFunc(vPoints),
                               utils.createCurveFunc(bPoints),
                               utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)

class BGRPortraCurveFilter(BGRCurveFilter):
    '''filter that applies portra-like curves to BGR'''
    def __init__(self, dtype=numpy.uint8):
        BGRCurveFilter.__init__(
            self, 
            vPoints=[(0,0), (23,20), (157,173), (255,255)],
            bPoints=[(0,0), (23,20), (231,228), (255,255)],
            gPoints=[(0,0), (23,20), (189,196), (255,255)],
            rPoints=[(0,0), (69,69), (213,218), (255,255)],
            dtype=dtype
        )