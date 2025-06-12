import cv2
from managers import WindowManager, CaptureManager

class cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        '''run the main loop'''
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                # TODO: filter the frame (chapter3)
                pass
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