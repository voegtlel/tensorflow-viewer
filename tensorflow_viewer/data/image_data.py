import traceback
from abc import abstractmethod

from PIL import Image
from PyQt5.QtCore import pyqtSignal, QMutex, QRunnable, QObject

from tensorflow_viewer.data.loader import PerStepEntry, ThreadPool
from tensorflow_viewer.data.utils import MutexLock, MutexUnlock
from tensorflow_viewer.utils import except_print


class ImageDataFutureSignals(QObject):
    done = pyqtSignal()
    data_ready = pyqtSignal(bytes)
    raw_data_ready = pyqtSignal(bytes, int, int, bool, str)


class ImageDataFuture(QRunnable):
    def __init__(self, event_entry, thread_pool):
        """
        Args:
            event_entry (ImageEntry):
            thread_pool (ThreadPool):
        """
        super(ImageDataFuture, self).__init__()
        self._event_entry = event_entry
        self._thread_pool = thread_pool
        self._image_data = None
        self._started = False
        self._finished = False
        self._cancelled = False
        self.setAutoDelete(False)
        self.signals = ImageDataFutureSignals()
        self._mutex = QMutex(QMutex.Recursive)

    @except_print
    def run(self):
        with MutexLock(self._mutex):
            try:
                if not self._cancelled:
                    with MutexUnlock(self._mutex):
                        data = self._event_entry.read_image_data()
                    self._image_data = data
                if not self._cancelled:
                    if self._image_data[0]:
                        self.signals.data_ready.emit(self._image_data[1])
                    else:
                        self.signals.raw_data_ready.emit(self._image_data[1], self._image_data[2], self._image_data[3], self._image_data[4], self._image_data[5])
            except BaseException as e:
                traceback.print_exc()
                print("Exception caught for {}".format(self._event_entry), flush=True)
                print(e)
            finally:
                if not self._cancelled:
                    self._finished = True
                    self.signals.done.emit()

    def start(self):
        with MutexLock(self._mutex):
            if not self._started:
                self._started = True
                self._thread_pool.start(self)

    def cancel(self):
        with MutexLock(self._mutex):
            if not self._cancelled and not self._finished:
                self._cancelled = True
                self._finished = True
                self._thread_pool.cancel(self)
                self.signals.done.emit()


class ImageEntry(PerStepEntry):
    def __init__(self, file, offset, tag, step, loader_id, thread_pool):
        """
        Creates an event entry.

        Args:
            file (LoaderFile): The events
            offset (int): Offset within the file
            tag (tuple[str|int]): The tag of this entry
            step (int): The step of this entry
            loader_id (tuple[int]): Id of the instantiating loader
            thread_pool (ThreadPool): The thread pool
        """
        super().__init__(file, offset, tag, step, loader_id)
        self._thread_pool = thread_pool
        #: :type: ImageDataFuture
        self._image_data_result = None

    @staticmethod
    def type():
        return "image"

    @abstractmethod
    def read_image_data(self):
        """
        Reads the image data from the record.

        Returns:
            True, bytearray: Compressed image data
            False, bytearray, int, int, bool, str: Raw image data, width, height, is_rgb, description
        """
        ...

    def read_image_data_thread(self):
        """
        Reads image data threaded.

        Returns:
            ImageDataFuture: The future returning the image data.
        """
        if self._image_data_result is None:
            self._image_data_result = ImageDataFuture(self, self._thread_pool)

            def on_done():
                self._image_data_result = None
            self._image_data_result.signals.done.connect(on_done)
        return self._image_data_result

    def close(self):
        super(ImageEntry, self).close()
        if self._image_data_result is not None:
            self._image_data_result.cancel()

    @staticmethod
    def image_to_result(img, message):
        if img.mode == 'L':
            channels = 1
            is_rgb = False
        elif img.mode == 'RGB':
            channels = 3
            is_rgb = True
        else:
            return True, None
        stride = img.width * channels
        if stride % 4 != 0:
            stride = (stride // 4 + 1) * 4
        img_bytes = img.tobytes('raw', 'RGB' if is_rgb else 'L', stride)

        return False, img_bytes, img.width, img.height, is_rgb, message + "\nSize: {}x{}x{}".format(img.height, img.width, channels)

    @classmethod
    def image_raw_to_result(cls, img_raw, height, width, message):
        channels = len(img_raw) // (height * width)
        if len(img_raw) != (height * width * channels):
            return True, None
        if channels == 1:
            is_rgb = False
        elif channels == 3:
            is_rgb = True
        else:
            return True, None
        stride = width * channels
        if stride % 4 != 0:
            return cls.image_to_result(Image.frombytes('L', (height, width), img_raw, 'raw', 'L', 1), message)
        return False, img_raw, width, height, is_rgb, message + "\nSize: {}x{}x{}".format(height, width, channels)
