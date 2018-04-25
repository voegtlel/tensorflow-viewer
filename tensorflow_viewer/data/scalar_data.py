import bisect

from tensorflow_viewer.data.loader import GlobalEntry


class ScalarEntry(GlobalEntry):
    def __init__(self, tag):
        """
        Creates a scalar entry.

        Args:
            tag (tuple[str|int]): The tag of this entry
        """
        super().__init__(tag)
        self._all_steps = []
        self._all_loader_ids = []
        self._steps = dict()
        self._data = dict()

    @staticmethod
    def is_per_step():
        return False

    @staticmethod
    def type():
        return "scalar"

    def steps(self, loader_id=None):
        if loader_id is None:
            return self._all_steps
        return tuple(self._steps[loader_id])

    def loader_ids(self):
        return self._all_loader_ids

    def close(self):
        super(ScalarEntry, self).close()
        self._steps = None
        self._data = None

    def get_data(self, loader_id):
        return tuple(self._data[loader_id])

    def add_data(self, step, value, loader_id):
        loader_id = (loader_id[0],)

        idx = bisect.bisect_left(self._all_steps, step)
        if not 0 <= idx < len(self._all_steps) or self._all_steps[idx] != step:
            self._all_steps.insert(idx, step)

        steps = self._steps.get(loader_id)
        if steps is None:
            steps = []
            data = []
            self._steps[loader_id] = steps
            self._data[loader_id] = data
            self._all_loader_ids.append(loader_id)
        else:
            data = self._data[loader_id]
        idx = bisect.bisect_right(steps, step)
        steps.insert(idx, step)
        data.insert(idx, value)
        self.signals.step_added.emit(idx, loader_id)
