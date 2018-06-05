from io import BytesIO

from PIL import Image, ImageMath
from tensorflow.core.example import example_pb2
from tensorflow.python.framework.errors_impl import DataLossError

from tensorflow_viewer.data.image_data import ImageEntry
from tensorflow_viewer.data.loader import AbstractLoader, loaders, LoaderFile


DEFAULT_MASK_PALETTE_DATA = (
    141, 211, 199,
    255, 255, 179,
    190, 186, 218,
    251, 128, 114,
    128, 177, 211,
    253, 180, 98,
    179, 222, 105,
    252, 205, 229,
    217, 217, 217,
    188, 128, 189,
    204, 235, 197,
    255, 237, 111,
)

DEFAULT_MASK_PALETTE_DATA_RAW = bytearray(
    list(DEFAULT_MASK_PALETTE_DATA) +
    [255] * (256*3 - len(DEFAULT_MASK_PALETTE_DATA))
)


class TFRecordEntryImage(ImageEntry):
    def __init__(self, file, offset, step, name, label, loader_id, thread_pool):
        """
        Creates an event entry.

        Args:
            file (LoaderFile): The events
            offset (int): Offset within the file
            step (int): The step of this entry
            loader_id (tuple[int]): Id of the instantiating loader
            thread_pool (ThreadPool): The thread pool
        """
        super(TFRecordEntryImage, self).__init__(file, offset, ("image",), step, loader_id, thread_pool)
        self.name = name
        self.label = label

    def read_image_data(self):
        """
        Reads the image data from the record.

        Returns:
            bytearray: Image data
        """
        example = self._file.read_example(self._offset)

        if 'height' in example.features.feature and 'width' in example.features.feature:
            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])

            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            info = "{}\nName: {}".format(
                self.tag_str(), self.name
            )
            if self.label is not None:
                info += "\nLabel: {}".format(self.label)

            if "image_raw" in example.features.feature:
                info += "\nCompressed: False"
                img_string = (example.features.feature['image_raw']
                    .bytes_list
                    .value[0])
                return self.image_raw_to_result(img_string, height, width, info)
            elif "image_compressed" in example.features.feature:
                info += "\nCompressed: True"
                byte_data = example.features.feature['image_compressed'].bytes_list.value[0]
                return self.image_to_result(Image.open(BytesIO(byte_data)), info)
        return True, None


class TFRecordEntryMask(ImageEntry):
    def __init__(self, file, offset, step, loader_id, mask_idx, name, thread_pool):
        """
        Creates an event entry.

        Args:
            file (LoaderFile): The events
            offset (int): Offset within the file
            step (int): The step of this entry
            loader_id (tuple[int]): Id of the instantiating loader
            mask_idx (int): Index of the requested mask
            thread_pool (ThreadPool): The thread pool
        """
        super(TFRecordEntryMask, self).__init__(file, offset, ("mask", mask_idx), step, loader_id, thread_pool)
        self._mask_idx = mask_idx
        self.name = name

    @staticmethod
    def read_num_masks(example):
        if 'height' in example.features.feature and 'width' in example.features.feature:
            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])
            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            if "mask_raw" in example.features.feature.keys():
                mask_string = (example.features.feature['mask_raw']
                    .bytes_list
                    .value[0])
                return len(mask_string) // (width * height)
            elif "mask_compressed" in example.features.feature.keys():
                mask_string = (example.features.feature['mask_compressed']
                    .bytes_list
                    .value[0])
                mask_data = Image.open(BytesIO(mask_string))
                return mask_data.height // height
        return 0

    def read_image_data(self):
        """
        Reads the image data from the record.

        Returns:
            bytearray: Image data
        """
        example = self._file.read_example(self._offset)

        if 'height' in example.features.feature and 'width' in example.features.feature:
            height = int(example.features.feature['height']
                         .int64_list
                         .value[0])
            width = int(example.features.feature['width']
                        .int64_list
                        .value[0])

            info = "{}\nName: {}".format(self.tag_str(), self.name)

            mask_channels = int(8)

            if "mask_raw" in example.features.feature.keys():
                info += "\nCompressed: False"
                mask_string = (example.features.feature['mask_raw']
                    .bytes_list
                    .value[0])
                img_size = width*height
                mask = mask_string[self._mask_idx*img_size:(self._mask_idx + 1)*img_size]

                mask_data = Image.frombytes('L', (height, width), mask, 'raw', 'L', 1)
                factor = 255 // mask_channels
                mask_data = ImageMath.eval("convert(a*{}, 'L')".format(factor), a=mask_data)
                return self.image_to_result(mask_data, info)
            elif "mask_compressed" in example.features.feature.keys():
                info += "\nCompressed: True"
                mask_string = (example.features.feature['mask_compressed']
                    .bytes_list
                    .value[0])
                mask_data = Image.open(BytesIO(mask_string))
                if mask_data.mode != 'L':
                    return True, None
                mask_data = mask_data.crop((0, self._mask_idx * height, width, height))

                mask_data.putpalette(DEFAULT_MASK_PALETTE_DATA_RAW)
                mask_data = mask_data.convert(mode='RGB')
                #factor = 255 // mask_channels
                #mask_data = ImageMath.eval("convert(a*{}, 'L')".format(factor), a=mask_data)
                return self.image_to_result(mask_data, info)
        return True, None


class TFRecordLoader(AbstractLoader):
    def __init__(self, path, id):
        super().__init__(path, id)
        self._iteration = 0
        self._file = LoaderFile(path)

    @staticmethod
    def applies_to(path):
        return ".tfrecords" in path

    def bytes_loaded(self):
        return self._file.offset()

    def bytes_total(self):
        return self._file.size()
    
    def key(self):
        return self._file.last_change()

    def _load_tfrecord(self, record, offset, target):
        example = example_pb2.Example.FromString(record)

        with target.lock():
            if 'identifier' in example.features.feature:
                name = example.features.feature['identifier'].bytes_list.value[0].decode()
            else:
                name = None
            if "label" in example.features.feature.keys():
                label = int(example.features.feature['label']
                            .int64_list
                            .value[0])
            else:
                label = None

            if 'height' in example.features.feature and 'width' in example.features.feature:
                if "image_raw" in example.features.feature or "image_compressed" in example.features.feature:
                    entry = TFRecordEntryImage(self._file, offset, self._iteration, name, label, self._id, target.thread_pool)
                    target.add_entry(entry)
                if "mask_raw" in example.features.feature or "mask_compressed" in example.features.feature:
                    num_masks = TFRecordEntryMask.read_num_masks(example)
                    for i in range(num_masks):
                        entry = TFRecordEntryMask(self._file, offset, self._iteration, self._id, 0, name, target.thread_pool)
                        target.add_entry(entry)

    def load(self, target):
        if not self._file.is_valid():
            # File was reset, cancel
            print("File was deleted, cancelling")
            target.delete_loader(self._id)
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
                    self._load_tfrecord(reader.record(), start_offset, target)
                    self._iteration += 1
                    self._file.set_offset(reader.offset())
                    target.next_iteration()
        except DataLossError as e:
            print(e)
        except BaseException as e:
            print(e)
            raise

        return True

    def __str__(self):
        return "TFRecordsLoader({})".format(self._path)

    def __repr__(self):
        return "TFRecordsLoader(" + repr(self._path) + ")"


loaders.append(TFRecordLoader)
