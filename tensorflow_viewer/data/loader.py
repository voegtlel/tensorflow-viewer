import bisect
import functools
import os
from abc import ABC, abstractmethod

from PyQt5.QtCore import QThread, pyqtSignal, QWaitCondition, QMutex, QThreadPool, QObject
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow, compat
from tensorflow.python.framework import errors
from tensorflow.python.lib.io.tf_record import TFRecordOptions

from tensorflow_viewer.data.utils import MutexLock
from tensorflow_viewer.utils import except_print


from distutils.version import StrictVersion
_new_tfrecord_interface = StrictVersion(tf.__version__) >= StrictVersion('1.8.0')


class TFRecordReader:
    def __init__(self, path, start_offset, options):
        compression_type = TFRecordOptions.get_compression_type_string(options)
        with errors.raise_exception_on_not_ok_status() as status:
            self._reader = pywrap_tensorflow.PyRecordReader_New(
                compat.as_bytes(path), start_offset, compat.as_bytes(compression_type), status
            )
        if self._reader is None:
            raise IOError("Could not open %s." % path)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if _new_tfrecord_interface:
                self._reader.GetNext()
            else:
                with errors.raise_exception_on_not_ok_status() as status:
                    self._reader.GetNext(status)
            return self
        except errors.OutOfRangeError:
            raise StopIteration

    def record(self):
        return self._reader.record()

    def offset(self):
        return self._reader.offset()

    def close(self):
        self._reader.Close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ThreadPool:
    def __init__(self, parent=None):
        self._thread_pool = QThreadPool(parent)
        self._mutex = QMutex(QMutex.Recursive)
        self._pending = []

    def start(self, runnable):
        with MutexLock(self._mutex):
            def on_done():
                with MutexLock(self._mutex):
                    self._pending.remove(runnable)
            runnable.signals.done.connect(on_done)
            self._pending.append(runnable)
            self._thread_pool.start(runnable)

    def cancel(self, runnable):
        with MutexLock(self._mutex):
            self._thread_pool.cancel(runnable)

    def stop(self):
        with MutexLock(self._mutex):
            self._thread_pool.clear()
            for pending in self._pending:
                pending.cancel()
        self._thread_pool.waitForDone()
        with MutexLock(self._mutex):
            self._thread_pool = None
            if self._pending:
                print("WARNING: Threads still pending:", self._pending)


class Entry(ABC):
    def __init__(self, tag):
        """
        Creates an entry.

        Args:
            tag (tuple[str|int]): The tag of this entry
        """
        self._tag = tag

    @staticmethod
    @abstractmethod
    def is_per_step():
        """
        Gets if the entry is separate per step. Otherwise it is for the whole file.

        Returns:
            bool: True if the entry is for a single step, otherwise for the whole file
        """
        ...

    @property
    def tag(self):
        return self._tag

    @staticmethod
    @abstractmethod
    def type():
        """
        Gets the type of this entry.
        Returns:
            ('image'|'scalar'): type of this entry
        """
        ...

    def tag_str(self):
        return "/".join(str(tag) for tag in self._tag)

    def __lt__(self, other):
        return self.tag < other.tag

    def close(self):
        pass

    def __str__(self):
        return "Entry(tag={})".format(repr(self.tag_str()))

    def __repr__(self):
        return self.__str__()


class PerStepEntry(Entry):
    def __init__(self, file, offset, tag, step, loader_id):
        """
        Creates a per step entry.

        Args:
            file (LoaderFile): The origin file
            offset (int|None): Offset within the file
            tag (tuple[str|int]): The tag of this entry
            step (int|None): The step of this entry
            loader_id (tuple[int]): Id of the instantiating loader
        """
        super().__init__(tag)
        self._file = file
        self._offset = offset
        self._step = step

    @staticmethod
    def is_per_step():
        return True

    @property
    def step(self):
        return self._step

    def __lt__(self, other):
        return self.step < other._step

    def close(self):
        self._file = None

    def __str__(self):
        return "Entry(tag={}, step={}, file={})".format(repr(self.tag_str()), repr(self._step), repr(self._file.path))


class ScalarDataSignals(QObject):
    step_added = pyqtSignal(int, tuple)


class GlobalEntry(Entry):
    def __init__(self, tag):
        """
        Creates a global entry.

        Args:
            tag (tuple[str|int]): The tag of this entry
        """
        super().__init__(tag)
        self.signals = ScalarDataSignals()

    @staticmethod
    def is_per_step():
        return False

    @abstractmethod
    def steps(self, loader_id=None):
        """
        Gets a list of steps.
        Args:
            loader_id (tuple[int]): Id of the loader

        Returns:
            tuple[int]: Steps
        """
        ...

    @abstractmethod
    def loader_ids(self):
        """
        Gets a list of loader ids.

        Returns:
            list[tuple[int]]: All loader ids in order of appearance
        """
        ...


class LoaderFile:
    def __init__(self, path):
        self.path = path
        self._last_offset = 0

    def is_valid(self):
        return os.path.exists(self.path) and self._last_offset <= self.size()

    def changed(self):
        return self._last_offset != self.size()

    def size(self):
        return os.stat(self.path).st_size

    def last_change(self):
        return os.stat(self.path).st_mtime

    def set_offset(self, offset):
        self._last_offset = offset

    def offset(self):
        return self._last_offset

    def get_reader(self, start_offset=None):
        if start_offset is None:
            start_offset = self._last_offset
        return TFRecordReader(self.path, start_offset, None)

    @functools.lru_cache(32)
    def read(self, offset):
        with self.get_reader(offset) as reader:
            next(reader)
            return reader.record()

    @functools.lru_cache(32)
    def read_event(self, offset):
        entry = self.read(offset)
        if entry is not None:
            return event_pb2.Event.FromString(entry)

    @functools.lru_cache(32)
    def read_example(self, offset):
        entry = self.read(offset)
        if entry is not None:
            return example_pb2.Example.FromString(entry)

    def __repr__(self):
        return "EventsLoaderFile({})".format(repr(self.path))

    def __str__(self):
        return "EventsLoaderFile({})".format(self.path)


class AbstractLoader(ABC):
    def __init__(self, path, id):
        """
        Args:
            path (str): Path to the file
            id (int): Id of the loader
        """
        self._path = path
        self._id = id

    @property
    def id(self):
        return self._id

    @abstractmethod
    def bytes_loaded(self):
        """
        Gets the progress of this loader.

        Returns:
            int: Processed bytes
        """
        ...

    @abstractmethod
    def bytes_total(self):
        """
        Gets the size of the contents of this loader.

        Returns:
            int: Size
        """
        ...

    @staticmethod
    @abstractmethod
    def applies_to(path):
        """
        Checks if the loader can process the given path.
        Args:
            path (str): Given path

        Returns:
            bool: This loader can process the given path.
        """
        ...

    @abstractmethod
    def load(self, target):
        """
        Loads data into the target.

        Args:
            target (DataLoader.DataLoaderTarget): The target to fill with records

        Returns:
            bool: False if the loader is closed, True otherwise
        """
        pass

    @abstractmethod
    def key(self):
        """
        Returns a key for sorting loaders.

        Returns:
            any: Key
        """
        ...

    def __str__(self):
        return "Loader for {}".format(self._path)


loaders = []


def get_loader(path, loader_id):
    """
    Args:
        path (str): Path to load
        loader_id (tuple[int]): Id for the loader

    Returns:
        AbstractLoader|None: The instantiated loader or None

    """
    for loader in loaders:
        if loader.applies_to(path):
            return loader(path, loader_id)


class DataLoader(QThread):
    load_clear = pyqtSignal()
    load_status = pyqtSignal(int, float)
    load_step = pyqtSignal(int, int)
    load_done = pyqtSignal()
    load_stop = pyqtSignal()
    load_tag = pyqtSignal(tuple, str)
    load_entry_global = pyqtSignal(tuple, str)

    class DataLoaderTarget:
        def __init__(self, loader):
            """
            Args:
                loader (DataLoader): The base loader
            """
            self._loader = loader

        def is_interruption_requested(self):
            """
            Checks if processing should interrupt.
            Returns:
                bool:
            """
            return self._loader.isInterruptionRequested()

        def lock(self):
            """
            Locks for being ready to add new records.
            Returns:
                MutexLock: The lock manager
            """
            return MutexLock(self._loader._mutex)

        def delete_loader(self, loader_id):
            """
            Deletes the given loader_id.
            Args:
                loader_id (tuple[int]):

            Returns:

            """
            # TODO: Implement
            ...

        def tag_to_path(self, tag):
            """
            Converts tag to path.

            Args:
                tag (str):

            Returns:
                tuple[str|int]: The path
            """
            return self._loader._tag_to_path(tag)

        def add_entry(self, entry):
            """
            Adds an entry.

            Args:
                entry (Entry): The entry to add.
            """
            self._loader._add_entry(entry)

        def get_global_entry(self, tag):
            """
            Gets an entry which is not per step.
            Args:
                tag (tuple[str|int]): Step

            Returns:
                Entry|None: The entry
            """
            return self._loader._global_tag_index.get(tag)

        @property
        def thread_pool(self):
            """
            Gets the thread pool.
            Returns:
                ThreadPool: Thread pool
            """
            return self._loader._thread_pool

        def next_iteration(self):
            """
            Notify for next iteration.
            """
            self._loader._finish_iteration()

    def _get_next_index(self):
        index = self._next_loader_index
        self._next_loader_index += 1
        return (index,)

    def __init__(self, paths, load_interval=2500, interactive_preload=False):
        super(DataLoader, self).__init__()
        self._load_interval = load_interval
        self._interactive_preload = interactive_preload
        self._next_loader_index = 0
        #: :type: list[AbstractLoader]
        self._loaders = [get_loader(path, self._get_next_index()) for path in paths]
        for loader, path in zip(self._loaders, paths):
            if loader is None:
                print("Cannot load", path)
        self._loaders = [loader for loader in self._loaders if loader is not None]
        self._loaders.sort(key=lambda loader: loader.key())
        self._mutex = QMutex(QMutex.Recursive)
        self._wait_mutex = QMutex()
        self._wait_cond = QWaitCondition()
        self._initial_load = True
        self._iteration = 0
        # Key is tag
        #: :type: dict[tuple[str|int], list[EventEntry]]
        self._tag_index = dict()
        # Key is tag
        #: :type: dict[tuple[str|int], EventEntry]
        self._global_tag_index = dict()
        #: :type: dict[str,tuple[str|int]]
        self._tag_path_index = dict()
        # Key is step, then tag
        #: :type: dict[int, dict[tuple[str|int], EventEntry]]
        self._step_index = dict()
        # All steps
        self._steps = []
        # All tags
        self._tags = []
        self._tag_types = []

        self._thread_pool = ThreadPool(self)
        self._pending_threads = []

        self._pending_steps = []

        self._loader_target = DataLoader.DataLoaderTarget(self)

    def add_load(self, path):
        self._initial_load = True
        loader = get_loader(path, self._get_next_index())
        if loader is None:
            print("Cannot load", path)
        else:
            with MutexLock(self._mutex):
                self._loaders.append(loader)

    def __del__(self):
        self.stop()

    def _tag_to_path(self, tag):
        """
        Args:
            tag (str):

        Returns:
            tuple[str|int]: The path
        """
        path = self._tag_path_index.get(tag)
        if path is None:
            if tag.endswith('/image'):
                tag = tag[:-6]
            i = len(tag) - 1
            path = []
            while i >= 0:
                if tag[i] == '/':
                    num = tag[i+1:]
                    if num.isdigit():
                        tag = tag[:i]
                        path.insert(0, int(num))
                    else:
                        break
                i -= 1
            path.insert(0, tag)
            self._tag_path_index[tag] = path = tuple(path)
        return path

    def _add_entry(self, entry):
        """
        Add an entry.

        Args:
            entry (Entry|PerStepEntry):
        """
        if entry.is_per_step():
            if entry.step not in self._step_index:
                self._step_index[entry.step] = dict()
            if entry.tag not in self._tag_index:
                self._tag_index[entry.tag] = list()
                self._tags.append(entry.tag)
                self._tag_types.append(entry.type())
                if self._interactive_preload or not self._initial_load:
                    print("Added tag:", entry.tag)
                    self.load_tag.emit(entry.tag, entry.type())
            self._step_index[entry.step][entry.tag] = entry
            bisect.insort_right(self._tag_index[entry.tag], entry)
            stepidx = bisect.bisect_left(self._steps, entry.step)
            if stepidx >= len(self._steps) or self._steps[stepidx] != entry.step:
                self._steps.insert(stepidx, entry.step)
                self._pending_steps.append((stepidx, self._iteration))
        else:
            assert entry.tag not in self._global_tag_index
            self._global_tag_index[entry.tag] = entry
            if self._interactive_preload or not self._initial_load:
                self.load_entry_global.emit(entry.tag, entry.type())

    def _finish_iteration(self):
        total_read = sum(loader.bytes_loaded() for loader in self._loaders)
        total = sum(loader.bytes_total() for loader in self._loaders)
        if self._initial_load:
            if self._iteration % 10 == 0:
                self.load_status.emit(self._iteration, total_read / total)
        else:
            print("Got additional iteration:", self._iteration)
            self.load_status.emit(self._iteration, 1.0)
        self._iteration += 1
        if self._interactive_preload or not self._initial_load:
            for stepidx, iteration in self._pending_steps:
                self.load_step.emit(stepidx, iteration)
        self._pending_steps.clear()

    def _load_files(self):
        invalid_loaders = []
        if self._initial_load:
            print("Loading", self._loaders)
        for loader in self._loaders:
            is_valid = loader.load(self._loader_target)
            if not is_valid:
                invalid_loaders.append(loader)
            if self.isInterruptionRequested():
                return False
        for loader in invalid_loaders:
            self._loaders.remove(loader)
        if not self._loaders:
            return False

        if self._initial_load:
            if not self._interactive_preload:
                for tag, tag_type in zip(self._tags, self._tag_types):
                    print("Added tag:", tag)
                    self.load_tag.emit(tag, tag_type)

                for entry in self._global_tag_index.values():
                    self.load_entry_global.emit(entry.tag, entry.type())

            print("Loaded file(s) initially")
            self.load_status.emit(self._iteration, 1.0)
            print("Load done")
            self.load_done.emit()
            self._initial_load = False
        return True

    @except_print
    def run(self):
        print("Thread started")
        try:
            while not self.isInterruptionRequested():
                with MutexLock(self._wait_mutex):
                    if not self._load_files():
                        break
                    if self._wait_cond.wait(self._wait_mutex, self._load_interval):
                        break
        except BaseException as e:
            print(e)
            raise
        finally:
            with MutexLock(self._mutex):
                self.load_stop.emit()
                if self._thread_pool is not None:
                    print("Waiting for thread pool")
                    self._thread_pool.stop()
                    self._thread_pool = None
        print("Thread exited")

    def reload(self):
        self.load_clear.emit()

    @property
    def initial_loaded(self):
        return self._initial_load

    @property
    def steps(self):
        """
        Gets all steps in ascending order.

        Returns:
            list[int]: All steps

        """
        with MutexLock(self._mutex):
            return list(self._steps)

    @property
    def tags(self):
        """
        Gets all tags.

        Returns:
            list[tuple[str|int]]: All tags

        """
        with MutexLock(self._mutex):
            return list(self._tags)

    def get_global_entry(self, tag):
        with MutexLock(self._mutex):
            return self._global_tag_index.get(tag)

    @property
    def tag_index(self):
        """
        Gets the index by tag.

        Returns:
            dict[tuple[str|int], list[EventEntry]]: The tag index
        """
        with MutexLock(self._mutex):
            return self._tag_index

    def stop(self):
        with MutexLock(self._mutex):
            self._wait_cond.wakeAll()
            self.requestInterruption()
        print("Waiting for finish")
        self.wait()
