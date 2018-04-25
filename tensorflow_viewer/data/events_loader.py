import os

from tensorflow_viewer.data.event_loader import EventLoader
from tensorflow_viewer.data.loader import AbstractLoader, loaders


class EventsLoader(AbstractLoader):
    def _make_next_sub_id(self):
        next_sub_id = self._next_sub_id
        self._next_sub_id += 1
        return self._id + (next_sub_id,)

    def __init__(self, path, id):
        super().__init__(path, id)

        self._next_sub_id = 0

        #: :type: list[data.event_loader.EventLoader]
        self._sub_loaders = [EventLoader(os.path.join(self._path, file), self._make_next_sub_id()) for file in os.listdir(self._path) if '.tfevents' in file]
        print("Loading", self._sub_loaders)

    @staticmethod
    def applies_to(path):
        return os.path.isdir(path) and any('.tfevents' in file for file in os.listdir(path))

    def bytes_loaded(self):
        return sum(sub_loader.bytes_loaded() for sub_loader in self._sub_loaders)

    def bytes_total(self):
        return sum(sub_loader.bytes_total() for sub_loader in self._sub_loaders)

    def key(self):
        return max(sub_loader.key() for sub_loader in self._sub_loaders)

    def load(self, target):
        if not os.path.isdir(self._path):
            print("Directory was deleted, cancelling")
            for sub_loader in self._sub_loaders:
                target.delete_loader(sub_loader.id)
            target.delete_loader(self._id)
            return False
        possible_files = set(os.path.join(self._path, file) for file in os.listdir(self._path) if '.tfevents' in file)
        possible_files.difference_update(sub_loader._path for sub_loader in self._sub_loaders)
        for new_file in possible_files:
            print("Load new file", new_file)
            self._sub_loaders.append(EventLoader(new_file, self._make_next_sub_id()))
        self._sub_loaders.sort(key=lambda sub_loader: sub_loader.key())
        removed_subloaders = []
        for sub_loader in self._sub_loaders:
            if not sub_loader.load(target):
                removed_subloaders.append(sub_loader)
            if target.is_interruption_requested():
                return False
        for sub_loader in removed_subloaders:
            self._sub_loaders.remove(sub_loader)
            target.delete_loader(sub_loader.id)
        return True

    def __str__(self):
        return "EventLoaders({})".format(self._sub_loaders)

    def __repr__(self):
        return "EventsLoader(" + repr(self._path) + ")"


loaders.append(EventsLoader)
