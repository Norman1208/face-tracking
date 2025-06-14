import cv2
import filters
from managers import WindowManager, CaptureManager

class cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = filters.BGRPortraCurveFilter()

    def run(self):
        '''run the main loop'''
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                # TODO: filter the frame 
                filters.strokeEdges(frame, frame)
                self._curveFilter.apply(frame, frame)
            self._captureManager.exitFrame() 
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        '''Handle a keypress.
        space -> take a screenshot
        tab -> start/stop recording a screencast
        escape -> quit'''
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingImage:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: #escape
            self._windowManager.destroyWindow()


if __name__ == "__main__":
    cameo().run()