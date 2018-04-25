from io import BytesIO

from PIL import Image
from tensorflow.core.util import event_pb2
from tensorflow.python.framework.errors_impl import DataLossError

from tensorflow_viewer.data.image_data import ImageEntry
from tensorflow_viewer.data.loader import AbstractLoader, loaders, LoaderFile
from tensorflow_viewer.data.scalar_data import ScalarEntry


class ImageEventEntry(ImageEntry):
    def __init__(self, file, offset, index, tag, step, loader_id, thread_pool):
        """
        Creates an image event entry.

        Args:
            file (LoaderFile): The events
            offset (int): Offset within the file
            index (int): Index of this entry within the file data
            tag (tuple[str|int]): The tag of this entry
            step (int): The step of this entry
            loader_id (tuple[int]): Id of the instantiating loader
            thread_pool (ThreadPool): The thread pool
        """
        super(ImageEventEntry, self).__init__(file, offset, tag, step, loader_id, thread_pool)
        self._index = index

    def read_image_data(self):
        event = self._file.read_event(self._offset)
        value = event.summary.value[self._index]
        byte_data = value.image.encoded_image_string
        return self.image_to_result(Image.open(BytesIO(byte_data)), self.tag_str())


class EventLoader(AbstractLoader):
    def __init__(self, path, id):
        super().__init__(path, id)

        self._file = LoaderFile(path)

    @staticmethod
    def applies_to(path):
        return ".tfevents" in path

    def bytes_loaded(self):
        return self._file.offset()

    def bytes_total(self):
        return self._file.size()

    def key(self):
        return self._file.last_change()

    def load(self, target):
        if not self._file.is_valid():
            # File was reset, cancel
            print("File was deleted, cancelling")
            return False
        elif not self._file.changed():
            # Nothing to do, still the same file
            return True
        try:
            with self._file.get_reader() as reader:
                for _ in reader:
                    if target.is_interruption_requested():
                        return False
                    start_offset = self._file.offset()
                    event = event_pb2.Event.FromString(reader.record())
                    with target.lock():
                        for idx, v in enumerate(event.summary.value):
                            if v.WhichOneof('value') == 'image':
                                tag = target.tag_to_path(v.tag)
                                entry = ImageEventEntry(self._file, start_offset, idx, tag, event.step, self._id, target.thread_pool)
                                target.add_entry(entry)
                            elif v.WhichOneof('value') == 'simple_value':
                                tag = target.tag_to_path(v.tag)
                                #: :type: ScalarEntry
                                entry = target.get_global_entry(tag)
                                if entry is None:
                                    entry = ScalarEntry(tag)
                                    entry.add_data(event.step, v.simple_value, self.id)
                                    target.add_entry(entry)
                                else:
                                    entry.add_data(event.step, v.simple_value, self.id)
                    self._file.set_offset(reader.offset())
                    target.next_iteration()
        except DataLossError as e:
            print(e)
        except BaseException as e:
            print(e)
            raise

        return True

    def __str__(self):
        return "EventLoader({})".format(self._path)

    def __repr__(self):
        return "EventLoader(" + repr(self._path) + ")"


loaders.append(EventLoader)
