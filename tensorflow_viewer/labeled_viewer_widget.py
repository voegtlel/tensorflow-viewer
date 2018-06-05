import bisect

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QWidget, QSizePolicy

from tensorflow_viewer.flow_layout import OrderedItem
from tensorflow_viewer.image_viewer_widget import ImageViewerWidget
from tensorflow_viewer.scalar_viewer_widget import ScalarViewerWidget
from tensorflow_viewer.step_widget import StepWidget
from tensorflow_viewer.utils import except_print


class StructuredTag:
    def __init__(self, tag_path, tag_index=1, is_global=False):
        self.tag_name = tag_path[0]
        self.tag_index = tag_index
        self.is_global = is_global
        if len(tag_path) > tag_index:
            key = tag_path[tag_index]
            #: :type: list[int]
            self.tag_values = [tag_path[tag_index]]
            #: :type: dict[int, StructuredTag]
            self.sub_tags = {key: StructuredTag(tag_path, tag_index + 1)}
            self.path = None
            self.steps = None
        else:
            self.tag_values = None
            self.sub_tags = None
            self.path = tag_path
            self.steps = []

    def add_tag(self, tag_path):
        assert self.tag_name == tag_path[0], "Invalid path {} for {}".format(self.tag_name, tag_path)
        assert len(tag_path) > self.tag_index, "Invalid path {} for {}".format(self.tag_name, tag_path)
        assert self.tag_values is not None
        assert self.sub_tags is not None
        tag_key = tag_path[self.tag_index]
        if tag_key in self.sub_tags:
            self.sub_tags[tag_key].add_tag(tag_path)
        else:
            bisect.insort_right(self.tag_values, tag_key)
            self.sub_tags[tag_key] = StructuredTag(tag_path, self.tag_index + 1)

    def set_steps(self, tag_path, steps):
        if len(tag_path) > self.tag_index:
            self.sub_tags[tag_path[self.tag_index]].set_steps(tag_path, steps)
        else:
            self.steps = steps

    def __str__(self):
        if self.path:
            return '/'.join(str(tag) for tag in self.path)
        return str(list(self.all_final_tags()))

    def all_final_tags(self):
        if self.sub_tags is None:
            yield self
        else:
            for tag in self.sub_tags.values():
                for final_tag in tag.all_final_tags():
                    yield final_tag


class LabeledViewerWidget(QWidget, OrderedItem):
    def __init__(self, parent=None, tag_path=(), tag_type=None):
        super(LabeledViewerWidget, self).__init__(parent)

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        sizePolicy.setHeightForWidth(True)
        sizePolicy.setHorizontalStretch(1)
        self.setSizePolicy(sizePolicy)

        self._layout = QVBoxLayout(self)

        #: :type: list[events.Entry]
        self._steps = []

        self._name = tag_path[0]

        self._selected_step_index = -1

        self._tag_label = QLabel(self)
        self._tag_label.setWordWrap(True)
        self._tag_label.setText(tag_path[0])
        self._layout.addWidget(self._tag_label)

        if tag_type == 'image':
            self._viewer = ImageViewerWidget(self)
        elif tag_type == 'scalar':
            self._viewer = ScalarViewerWidget(self)
        else:
            raise ValueError("Unsupported tag type")
        self._layout.addWidget(self._viewer, 1)

        self._step_widget = StepWidget(self)
        self._step_widget.setVisible(False)
        self._step_widget.stepChanged.connect(self._step_changed)
        self._layout.addWidget(self._step_widget)

        #: :type: list[StepWidget]
        self._tag_widgets = []
        #: :type: StructuredTag
        self._tags = StructuredTag(tag_path)

        if len(tag_path) > 1:
            self._get_tag_widget(0).setVisible(True)
            self._set_tag_path(tag_path)

        self.setLayout(self._layout)

    def set_preferred_size(self, size):
        self._viewer.set_preferred_size(size)

    def reset_transform(self):
        self._viewer.reset_transform()

    def _get_tag_widget(self, index):
        while index >= len(self._tag_widgets):
            tag_widget = StepWidget(self)
            tag_widget.setVisible(False)
            tag_widget.stepChanged.connect(self._tag_step_changed)

            self._tag_widgets.append(tag_widget)
            self._layout.addWidget(tag_widget)
        return self._tag_widgets[index]

    def add_tag_path(self, tag_path):
        selected_tag = self._get_selected_tag()
        self._tags.add_tag(tag_path)
        if selected_tag is None:
            self._set_tag_path(tag_path)
        else:
            self._set_tag_path(selected_tag)

    @property
    def orderTag(self):
        return self._name

    def _set_tag_path(self, tag_path):
        """
        Args:
            tag_path (tuple[str|int]): Path
        """
        tags = self._tags
        if tag_path is not None:
            for idx, key in enumerate(tag_path[1:]):
                widget = self._get_tag_widget(idx)
                widget.setSteps(tags.tag_values)
                widget.setStep(key)
                widget.setVisible(True)
                tags = tags.sub_tags[key]
            for widget in self._tag_widgets[len(tag_path):]:
                widget.setVisible(False)
            self._set_steps(tags.steps, True)
            self._tag_label.setText(str(tags))
        else:
            if len(self._tag_widgets) > 0:
                for widget in self._tag_widgets[1:]:
                    widget.setVisible(False)
            self._set_steps(None, True)
            self._tag_label.setText(self._tags.tag_name)

    def set_step(self, step):
        """
        Args:
            step (int): The step to assign
        """
        self._step_widget.setStep(step)

    def set_allow_steps(self, allow_steps):
        self._step_widget.setVisible(allow_steps)

    @except_print
    def _global_step_added(self, new_index, loader_id):
        self._step_widget.setSteps(self._steps[0].steps())

    @except_print
    def _set_steps(self, steps, force_update):
        if len(self._steps) == 1 and not self._steps[0].is_per_step():
            self._steps[0].signals.step_added.disconnect(self._global_step_added)
        if steps is None:
            self._steps = []
            self._step_widget.setSteps(None, force_update)
        elif len(steps) == 1 and not steps[0].is_per_step():
            has_change = (
                force_update
                or self._selected_step_index != 0
                or len(self._steps) != 1
                or len(steps) != 1
                or self._steps[0] != steps[0]
            )
            self._steps = steps
            steps[0].signals.step_added.connect(self._global_step_added)
            self._step_widget.setSteps(steps[0].steps(), has_change)
        else:
            has_change = (
                force_update
                or self._selected_step_index == -1
                or len(self._steps) <= self._selected_step_index
                or len(steps) <= self._selected_step_index
                or self._steps[self._selected_step_index] != steps[self._selected_step_index]
            )
            self._steps = steps
            self._step_widget.setSteps([step.step for step in steps], has_change)

    @except_print
    def set_steps(self, steps):
        """
        Sets the data array when the local slider is enabled

        Args:
            steps (dict[tuple[str|int], list[data.loader.Entry]]): List of entries
        """
        selected_tag = self._get_selected_tag()
        for tag in self._tags.all_final_tags():
            tag_steps = steps.get(tag.path)
            if tag_steps is not None:
                if selected_tag == tag.path:
                    self._set_steps(tag_steps, False)
                tag.steps = tag_steps

    @except_print
    def set_global_step(self, step):
        """
        Sets the data array when the local slider is enabled

        Args:
            step (data.loader.Entry): Step to update
        """
        self._tags.set_steps(step.tag, [step])

    def set_initial_loading(self, initial_loading):
        self._viewer.set_initial_loading(initial_loading)

    def set_interactive_preload(self, interactive_preload):
        self._viewer.set_interactive_preload(interactive_preload)

    def set_lock_step_end(self, lock):
        self._step_widget.set_lock_step_end(lock)

    def set_smoothing(self, smoothing):
        self._viewer.set_smoothing(smoothing)

    def _get_selected_tag(self):
        tags = self._tags
        for widget in self._tag_widgets:
            step = widget.getSelectedStep()
            if step is None:
                return None
            if tags.sub_tags is not None:
                tags = tags.sub_tags[step]
            else:
                return None
        return tags.path

    @except_print
    def _tag_step_changed(self, step, step_index):
        selected_tag = self._get_selected_tag()
        self._set_tag_path(selected_tag)

    @except_print
    def _step_changed(self, step, index):
        self._selected_step_index = index
        if hasattr(self._viewer, 'set_step_data'):
            if index != -1 and index < len(self._steps):
                assert self._steps[index].step == step
                self._viewer.set_step_data(self._steps[index])
            else:
                self._viewer.reset()
        elif hasattr(self._viewer, 'set_global_data'):
            if index != -1 and len(self._steps) > 0:
                assert len(self._steps) == 1, len(self._steps)
                self._viewer.set_global_data(self._steps[0], step)

    def save_file(self, path):
        self._viewer.save_file(path)
