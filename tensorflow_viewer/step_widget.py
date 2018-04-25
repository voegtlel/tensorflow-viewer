import bisect

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QHBoxLayout, QSlider, QLabel, QWidget

from tensorflow_viewer.utils import except_print


class StepWidget(QWidget):
    stepChanged = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super(StepWidget, self).__init__(parent)

        layout = QHBoxLayout(self)
        self._step_slider = QSlider(Qt.Horizontal, self)
        self._step_slider.valueChanged.connect(self._value_changed)
        self._step_slider.setPageStep(1)
        self._step_slider.setSingleStep(1)
        layout.addWidget(self._step_slider, 1)
        self._step_label = QLabel(self)
        layout.addWidget(self._step_label)
        self.setLayout(layout)

        self._selected = None

        self._steps = []

        self._lock_step_end = False

        self._update_step_label()

    def setSteps(self, data, force_update=False):
        """
        Sets the steps.

        Args:
            data (list[int]|None): List of steps
        """
        if not data:
            self._steps = []
            self._step_slider.setMaximum(0)
            self._step_slider.setValue(0)
        else:
            current_index = self._step_slider.value()
            has_new_selection = (
                force_update
                or current_index >= len(data)
                or current_index >= len(self._steps)
                or self._steps[current_index] != data[current_index]
            )

            should_set_initial = not self._steps
            self._steps = data
            had_max_selected = (
                self._lock_step_end
                and self._step_slider.value() == self._step_slider.maximum()
                and self._step_slider.value() != 0
            )
            self._step_slider.setMaximum(len(self._steps) - 1)
            if had_max_selected:
                if self._step_slider.value() != self._step_slider.maximum():
                    self._step_slider.setValue(self._step_slider.maximum())
                elif has_new_selection:
                    self._value_changed(self._step_slider.value())
            elif should_set_initial:
                self._value_changed(0)
            else:
                self._update_step_label()
                if has_new_selection:
                    self._value_changed(self._step_slider.value())

    def setStep(self, step):
        if step is None or step == -1:
            self._selected = None
            self.stepChanged.emit(-1, -1)
        else:
            idx = bisect.bisect_left(self._steps, step)
            self._step_slider.setValue(idx)
            if idx >= len(self._steps) or self._steps[idx] != step:
                self._selected = None
                self.stepChanged.emit(-1, -1)

    def set_lock_step_end(self, lock):
        self._lock_step_end = lock

    def getSelectedStep(self):
        index = self._step_slider.value()
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    @except_print
    def _value_changed(self, value):
        if value < len(self._steps):
            step = self._steps[value]
            self.stepChanged.emit(step, value)
            self._selected = step
        else:
            self.stepChanged.emit(-1, -1)
            self._selected = None
        self._update_step_label()

    def _update_step_label(self):
        step_count = self._steps[-1] if self._steps else 0
        if self._selected is None:
            self._step_label.setText("None/{}".format(step_count))
        else:
            self._step_label.setText("{}/{}".format(self._selected, step_count))

