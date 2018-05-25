import os

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QWidget

from tensorflow_viewer.flow_layout import FlowLayout
from tensorflow_viewer.labeled_viewer_widget import LabeledViewerWidget


class EventWidget(QWidget):
    def __init__(self, parent=None):
        super(EventWidget, self).__init__(parent)

        #: :type: dict[str, LabeledViewerWidget]
        self._tag_viewer = dict()

        #: :type: dict[str, LabeledViewerWidget]
        self._tag_viewer_global = dict()

        self._viewer_preferred_size = QSize(200, 200)

        self._interactive_preload = False

        self.layout = FlowLayout(self)
        self.setLayout(self.layout)

        self._allow_step_select = False
        self._lock_step_end = False
        self._smoothing = 1.0

    def clear(self):
        for tag, viewer in self._tag_viewer.items():
            self.layout.removeWidget(viewer)
            viewer.setParent(None)
            viewer.deleteLater()
        self._tag_viewer.clear()
        self._lock_step_end = False

    def set_lock_step_end(self, lock):
        self._lock_step_end = lock
        for viewer in self._tag_viewer.values():
            viewer.set_lock_step_end(lock)

    def set_events(self, events):
        """
        Args:
            events (dict[tuple[str|int], list[data.loader.Entry]]): All events
        """
        for tag, viewer in self._tag_viewer.items():
            viewer.set_steps(events)

    def add_tag(self, tag, tag_type):
        """
        Args:
            tag (tuple[str|int]):
            tag_type (str):
        Returns:
            LabeledViewerWidget: The viewer
        """
        viewer = self._tag_viewer.get(tag[0])
        if viewer is None:
            viewer = LabeledViewerWidget(self, tag, tag_type)
            viewer.set_preferred_size(self._viewer_preferred_size)
            viewer.set_allow_steps(self._allow_step_select)
            viewer.set_smoothing(self._smoothing)
            viewer.set_interactive_preload(self._interactive_preload)
            self.layout.addWidget(viewer)
            self._tag_viewer[tag[0]] = viewer
        else:
            viewer.add_tag_path(tag)
        return viewer

    def add_global_entry(self, entry):
        """
        Args:
            entry (Entry):
        """
        if entry.tag not in self._tag_viewer:
            viewer = self.add_tag(entry.tag, entry.type())
            viewer.set_steps({entry.tag: [entry]})

    def set_allow_step_select(self, allow):
        self._allow_step_select = allow
        for viewer in self._tag_viewer.values():
            viewer.set_allow_steps(allow)

    def set_smoothing(self, smoothing):
        self._smoothing = smoothing
        for viewer in self._tag_viewer.values():
            viewer.set_smoothing(smoothing)

    def set_viewer_preferred_size(self, size):
        self._viewer_preferred_size = size
        for viewer in self._tag_viewer.values():
            viewer.set_preferred_size(self._viewer_preferred_size)

    def reset_views(self):
        for viewer in self._tag_viewer.values():
            viewer.reset_transform()

    def set_step(self, step):
        for viewer in self._tag_viewer.values():
            viewer.set_step(step)

    def set_initial_loading(self, initial_loading):
        for viewer in self._tag_viewer.values():
            viewer.set_initial_loading(initial_loading)

    def set_interactive_preload(self, interactive_preload):
        self._interactive_preload = interactive_preload
        for viewer in self._tag_viewer.values():
            viewer.set_interactive_preload(interactive_preload)

    def save_files(self, parent_path):
        for tag, viewer in self._tag_viewer.items():
            viewer.save_file(os.path.join(parent_path, tag.replace('/', '_') + ".png"))

    def set_tags_visible(self, tags, visible):
        tag_keys = set(tag[0] for tag in tags)
        for tag_key in tag_keys:
            viewer = self._tag_viewer.get(tag_key)
            if viewer is not None:
                viewer.setVisible(visible)
        self.layout.update()
