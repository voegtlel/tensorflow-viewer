from PyQt5.QtCore import QMutex


class MutexLock:
    def __init__(self, mutex):
        """
        Args:
            mutex (QMutex):
        """
        self._mutex = mutex

    def __enter__(self):
        self._mutex.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mutex.unlock()


class MutexUnlock:
    def __init__(self, mutex):
        """
        Args:
            mutex (QMutex):
        """
        self._mutex = mutex

    def __enter__(self):
        self._mutex.unlock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mutex.lock()