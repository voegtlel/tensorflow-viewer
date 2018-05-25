import math
from typing import Union

import numpy as np
from PyQt5.QtCore import QPointF, QSize, QEvent, QObject, Qt, QRectF
from PyQt5.QtGui import QPaintEvent, QPainter, QTransform, QMouseEvent, \
    QWheelEvent, QPalette, QKeyEvent, QPainterPath, QPen, QBrush, QColor
from PyQt5.QtWidgets import QWidget, QFileDialog

from utils import except_print


class GridMapper:
    def __init__(self):
        self.value_min = 0.0
        self.value_range = 0.0
        self.multiplier = 0.0
        self.grid_start = 0.0
        self.view_min = 0.0
        self.view_max = 0.0

    def set(self, view_min, view_max, value_min, value_range):
        self.value_min = value_min
        self.value_range = value_range
        view_min = view_min * value_range + value_min
        view_max = view_max * value_range + value_min
        view_range = view_max - view_min

        next_10 = 1.0
        if view_range >= 1.0:
            while next_10 < view_range:
                next_10 *= 10.0
        else:
            while next_10 > view_range:
                next_10 *= 0.1
            next_10 *= 10

        if view_range > next_10 * 0.5:
            self.multiplier = 0.2 * next_10
        elif view_range > next_10 * 0.2:
            self.multiplier = 0.1 * next_10
        else:
            self.multiplier = 0.05 * next_10
        self.grid_start = round(view_min / self.multiplier) - 1

    def values(self):
        for val in range(7):
            value = (self.grid_start + val) * self.multiplier
            yield value, (value - self.value_min) / self.value_range


class ScalarDisplayData:
    def __init__(self, entry, loader_id):
        """

        Args:
            entry (data.scalar_data.ScalarEntry):
        """
        self._entry = entry
        self.loader_id = loader_id

        self._data = None
        self.data_path = None

        self._smooth_data = None
        self.smooth_data_path = None

    def update_range(self, global_min, global_range):
        self._data = np.array((self._entry.steps(self.loader_id), self._entry.get_data(self.loader_id)),
                              dtype=np.float32)

        min_changed = False

        if self._data.shape[1] > 1:
            local_min = np.min(self._data, axis=1)
            data_max = np.max(self._data, axis=1)
            local_min[1] = min(local_min[1], 0.0)
            data_max[1] = max(data_max[1], 0.0)
            local_range = data_max - local_min

            if local_range[0] <= 0.0 and global_range[0] <= 0.0:
                local_range[0] = 1.0
            if local_range[1] <= 0.0 and global_range[1] <= 0.0:
                local_range[1] = 1.0
            local_min -= local_range * 0.05
            local_range *= 1.1
            data_max[0] = local_min[0] + local_range[0]
            data_max[1] = local_min[1] + local_range[1]

            if (
                global_min[0] > local_min[0]
                or global_min[1] > local_min[1]
                or global_min[0] + global_range[0] < data_max[0]
                or global_min[1] + global_range[1] < data_max[1]
            ):
                min_changed = True
            local_min = np.minimum(local_min, global_min)
            data_max = np.maximum(data_max, global_min + global_range)
            local_range = data_max - local_min
            global_min[0] = local_min[0]
            global_min[1] = local_min[1]
            global_range[0] = local_range[0]
            global_range[1] = local_range[1]
        return min_changed

    def update(self, smoothing_kernel, global_min, global_range):
        if self._data.shape[1] > 1:
            self.data_path = self._data_to_path(self._data, global_min, global_range)
            if smoothing_kernel is not None:
                self._do_smooth_data(smoothing_kernel)
                self.smooth_data_path = self._data_to_path(self._smooth_data, global_min, global_range)
            else:
                self._smooth_data = self._data
                self.smooth_data_path = None
        else:
            self.data_path = None
            self.smooth_data_path = None

    def _data_to_path(self, data, global_min, global_range):
        transformed_data = (data - global_min[:, np.newaxis]) / global_range[:, np.newaxis]
        path = QPainterPath()
        path.moveTo(transformed_data[0, 0], transformed_data[1, 0])
        for i in range(transformed_data.shape[1]):
            path.lineTo(transformed_data[0, i], transformed_data[1, i])
        return path

    def _do_smooth_data(self, smoothing_kernel):
        self._smooth_data = np.convolve(
            np.pad(self._data[1, :], len(smoothing_kernel) // 2, mode='edge'),
            smoothing_kernel, mode='valid'
        )
        assert self._smooth_data.shape[0] == self._data.shape[1], (self._smooth_data.shape, self._data.shape)
        self._smooth_data = np.stack((self._data[0, :], self._smooth_data))
        assert self._smooth_data.shape == self._data.shape, (self._smooth_data.shape, self._data.shape)

    def snap(self, mouse_pos, snap_cursor, global_min, global_range):
        cursor_pos_x = mouse_pos.x() * global_range[0] + global_min[0]
        cursor_pos_y = mouse_pos.y() * global_range[1] + global_min[1]

        is_within = (
            self._data is not None
            and self._data.shape[1] > 0
            and self._data[0, 0] <= cursor_pos_x <= self._data[0, -1]
        )

        if snap_cursor:
            # Snap cursor to data
            if self._data is not None and self._data.shape[1] > 0:
                idx = np.searchsorted(self._data[0, :], cursor_pos_x)
                idx = min(max(0, idx), self._data.shape[1] - 1)
                prev_idx = min(max(0, idx - 1), self._data.shape[1] - 1)
                if abs(cursor_pos_x - self._data[0, idx]) > abs(cursor_pos_x - self._data[0, prev_idx]):
                    idx = prev_idx
                cursor_pos_x = self._data[0, idx]
                if self._smooth_data is not None:
                    cursor_pos_y = self._smooth_data[1, idx]
                else:
                    cursor_pos_y = self._data[1, idx]

        # Print cross
        cursor_x = (cursor_pos_x - global_min[0]) / global_range[0]
        cursor_y = (cursor_pos_y - global_min[1]) / global_range[1]
        return cursor_x, cursor_y, cursor_pos_x, cursor_pos_y, is_within


class ScalarViewerWidget(QWidget):
    def __init__(self, parent=None):
        super(ScalarViewerWidget, self).__init__(parent)

        self._interactive_preload = False

        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        self.offset = QPointF(0.5, 0.5)

        self._preferred_size = QSize(200, 200)

        self._step = 0

        self._draggingView = False
        self._drag_last = None
        self._draggingArea = False
        self._drag_start = None

        self.setBackgroundRole(QPalette.Dark)
        self.installEventFilter(self)
        self.setFocusPolicy(Qt.WheelFocus)

        self._pens_data = []
        self._pens_data_orig = []

        self._pen_marker = QPen(QBrush(Qt.black), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
        self._pen_marker.setCosmetic(True)

        self._pen_grid = QPen(QBrush(Qt.lightGray), 0.5, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
        self._pen_grid.setCosmetic(True)
        self._pen_grid_zero = QPen(QBrush(Qt.lightGray), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
        self._pen_grid_zero.setCosmetic(True)
        self._pen_grid_label = QPen(QBrush(QColor.fromRgb(220, 220, 220)), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
        self._pen_grid_label.setCosmetic(True)

        self._pen_highlight = QPen(QBrush(Qt.blue), 1.0, Qt.DotLine, Qt.SquareCap, Qt.BevelJoin)
        self._pen_highlight.setCosmetic(True)

        self._grid_mapper_x = GridMapper()
        self._grid_mapper_y = GridMapper()

        #: :type: QTransform
        self._transform = None
        #: :type: QTransform
        self._transform_inv = None
        self._last_width = None
        self._last_height = None

        #: :type: list[ScalarDisplayData]
        self._data = []
        self._scalar_data = None

        self._min = np.array((0.0, 0.0), dtype=np.float32)
        self._range = np.array((0.0, 0.0), dtype=np.float32)

        #: :type: QPointF
        self._cursor_pos = None
        self._cursor_snap = True

        self._smoothing = 1.0
        #: :type: np.array
        self._smoothing_kernel = None

        self._initial_loading = True

        self.setMouseTracking(True)

    def _get_pens(self):
        if len(self._data) > len(self._pens_data):
            self._pens_data = tuple(
                QPen(QBrush(QColor.fromHslF(i / len(self._data), 1.0, 0.5)), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
                for i in range(len(self._data))
            )
            self._pens_data_orig = tuple(
                QPen(QBrush(QColor.fromHslF(i / len(self._data), 0.5, 0.5)), 1.0, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin)
                for i in range(len(self._data))
            )
            for pen in self._pens_data:
                pen.setCosmetic(True)
            for pen in self._pens_data_orig:
                pen.setCosmetic(True)
        return self._pens_data, self._pens_data_orig

    @except_print(no_wrap=True)
    def sizeHint(self):
        return self._preferred_size

    def make_smoothing_kernel(self):
        count = int(math.floor(self._smoothing * 3))
        x = np.arange(-count, count+1, dtype=np.float32)
        kernel = np.exp(-0.5 * x * x / self._smoothing)
        kernel /= np.sum(kernel)
        self._smoothing_kernel = kernel

    def make_transform(self):
        self._transform = QTransform()
        self._last_width = self.width()
        self._last_height = self.height()
        # Center in widget
        self._transform.translate(self._last_width * 0.5, self._last_height * 0.5)
        self._transform.scale(self._last_width, -self._last_height)
        self._transform.scale(self.scale_factor_x, self.scale_factor_y)
        self._transform.translate(-self.offset.x(), -self.offset.y())
        self._transform_inv, _ = self._transform.inverted()
        return self._transform

    @except_print(no_wrap=True)
    def paintEvent(self, paint_event: QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.eraseRect(0, 0, self.width(), self.height())
        if any(data.data_path is not None for data in self._data):
            # Transform by parent transformation
            if self._last_width != self.width() or self._last_height != self.height():
                self._transform = None
            if self._transform is None:
                self.make_transform()

            start = QPointF()
            end = QPointF()

            if self._transform_inv is not None:
                view_rect = self._transform_inv.mapRect(QRectF(0.0, 0.0, self.width(), self.height()))
                #QRectF(inverted_transform.map(QPointF(0.0, 0.0)), inverted_transform.map(QPointF(self.width(), self.height())))

                self._grid_mapper_x.set(view_rect.left(), view_rect.right(), self._min[0], self._range[0])
                self._grid_mapper_y.set(view_rect.top(), view_rect.bottom(), self._min[1], self._range[1])
            else:
                view_rect = None

            painter.setTransform(self._transform)

            # Draw grid
            if self._transform_inv is not None:

                # Draw 0 lines
                painter.setPen(self._pen_grid_zero)

                # x zero line
                start.setY(view_rect.top())
                end.setY(view_rect.bottom())
                start.setX((-self._min[0]) / self._range[0])
                end.setX((-self._min[0]) / self._range[0])
                painter.drawLine(start, end)

                # y zero line
                start.setX(view_rect.left())
                end.setX(view_rect.right())
                start.setY((-self._min[1]) / self._range[1])
                end.setY((-self._min[1]) / self._range[1])
                painter.drawLine(start, end)

                # Draw all other grid lines
                # x lines
                start.setY(view_rect.top())
                end.setY(view_rect.bottom())
                painter.setPen(self._pen_grid)
                for value, view_pos in self._grid_mapper_x.values():
                    start.setX(view_pos)
                    end.setX(view_pos)
                    painter.drawLine(start, end)

                # y lines
                start.setX(view_rect.left())
                end.setX(view_rect.right())
                painter.setPen(self._pen_grid)
                for value, view_pos in self._grid_mapper_y.values():
                    start.setY(view_pos)
                    end.setY(view_pos)
                    painter.drawLine(start, end)

                painter.setPen(self._pen_grid_label)
                # Draw text for x lines
                start.setY(view_rect.top())
                end.setY(view_rect.bottom())
                painter.resetTransform()
                for value, view_pos in self._grid_mapper_x.values():
                    start.setX(view_pos)
                    startMapped = self._transform.map(start)
                    painter.drawText(startMapped, "{:.3g}".format(value))
                painter.setTransform(self._transform)
                # Draw text for y lines
                start.setX(view_rect.left())
                end.setX(view_rect.right())
                painter.resetTransform()
                for value, view_pos in self._grid_mapper_y.values():
                    start.setY(view_pos)
                    startMapped = self._transform.map(start)
                    painter.drawText(startMapped, "{:.3g}".format(value))
                painter.setTransform(self._transform)

            for i, (data, pen_data, pen_data_orig) in enumerate(zip(self._data, *self._get_pens())):
                if data.smooth_data_path:
                    # Draw data
                    painter.setPen(pen_data_orig)
                    painter.drawPath(data.data_path)

                    painter.setPen(pen_data)
                    painter.drawPath(data.smooth_data_path)
                elif data.data_path:
                    # Draw data
                    painter.setPen(pen_data)
                    painter.drawPath(data.data_path)

            # Draw step cursor
            painter.setRenderHint(QPainter.Antialiasing, True)
            if self._transform_inv is not None:
                painter.setPen(self._pen_marker)
                if self._cursor_pos is not None:
                    # Convert to data space
                    mouse_pos = self._transform_inv.map(self._cursor_pos)

                    snaps = tuple(data.snap(mouse_pos, self._cursor_snap, self._min, self._range) for data in self._data)
                    closest_idx = None
                    y_cmp = None
                    for i, (cursor_x, cursor_y, cursor_pos_x, cursor_pos_y, is_within) in enumerate(snaps):
                        if is_within:
                            dist = abs(cursor_y - mouse_pos.y())
                            if y_cmp is None or dist < y_cmp:
                                y_cmp = dist
                                closest_idx = i
                    if closest_idx is None:
                        x_cmp = None
                        for i, (cursor_x, cursor_y, cursor_pos_x, cursor_pos_y, is_within) in enumerate(snaps):
                            dist_x = abs(cursor_x - mouse_pos.x())
                            dist_y = abs(cursor_y - mouse_pos.y())
                            if x_cmp is None or dist_x < x_cmp:
                                x_cmp = dist_x
                                y_cmp = dist_y
                                closest_idx = i
                            elif dist_x == x_cmp and dist_y < y_cmp:
                                x_cmp = dist_x
                                y_cmp = dist_y
                                closest_idx = i
                    if closest_idx is not None:
                        cursor_x, cursor_y, cursor_pos_x, cursor_pos_y, _ = snaps[closest_idx]

                        # Vertical line
                        start.setX(cursor_x)
                        start.setY(view_rect.top())
                        end.setX(cursor_x)
                        end.setY(view_rect.bottom())
                        painter.drawLine(start, end)
                        # Horizontal line
                        start.setX(view_rect.left())
                        start.setY(cursor_y)
                        end.setX(view_rect.right())
                        end.setY(cursor_y)
                        painter.drawLine(start, end)

                        # Print text
                        painter.resetTransform()
                        start.setX(cursor_x)
                        start.setY(cursor_y)
                        startMapped = self._transform.map(start)
                        text = "{:.3g},{:.3g}".format(cursor_pos_x, cursor_pos_y)
                        if startMapped.x() > self.width() / 2:
                            textRect = painter.boundingRect(QRectF(), 0, text)
                            startMapped.setX(startMapped.x() - textRect.width())
                        painter.drawText(startMapped, text)
                else:
                    # Print step
                    step_x = (self._step - self._min[0]) / self._range[0]
                    start = QPointF(step_x, view_rect.top())
                    end = QPointF(step_x, view_rect.bottom())
                    painter.drawLine(start, end)

        if self._draggingArea:
            painter.resetTransform()
            painter.setPen(self._pen_highlight)
            painter.drawRect(
                self._drag_start.x(),
                self._drag_start.y(),
                self._drag_last.x() - self._drag_start.x(),
                self._drag_last.y() - self._drag_start.y()
            )

        painter.end()

    @except_print(no_wrap=True)
    def eventFilter(self, object: QObject, event: Union[QEvent, QMouseEvent, QWheelEvent, QKeyEvent]):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._draggingArea = True
                self._drag_start = event.pos()
                self._drag_last = self._drag_start
                event.accept()
                self.update()
                return True
            elif event.button() == Qt.RightButton:
                self._draggingView = True
                self._drag_last = event.pos()
                event.accept()
                self.update()
                return True
        elif event.type() == QEvent.MouseButtonDblClick:
            self.reset_transform()
            event.accept()
            return True
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self._draggingArea = False

                x_min = min(self._drag_start.x(), self._drag_last.x())
                x_max = max(self._drag_start.x(), self._drag_last.x())
                y_min = min(self._drag_start.y(), self._drag_last.y())
                y_max = max(self._drag_start.y(), self._drag_last.y())

                if x_max > x_min and y_max > y_min:
                    if self._transform is None:
                        self.make_transform()

                    selected_rect = self._transform_inv.mapRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
                    view_rect = self._transform_inv.mapRect(QRectF(0.0, 0.0, self.width(), self.height()))

                    center_offset = selected_rect.center() - view_rect.center()
                    self.offset.setX(self.offset.x() + center_offset.x())
                    self.offset.setY(self.offset.y() + center_offset.y())

                    self.scale_factor_x *= view_rect.width() / selected_rect.width()
                    self.scale_factor_y *= view_rect.height() / selected_rect.height()

                    self.make_transform()

                    self._transform = None

                event.accept()
                self.update()
                return True
            elif event.button() == Qt.RightButton:
                self._draggingView = False
                event.accept()
                self.update()
                return True
        elif event.type() == QEvent.Leave:
            self._cursor_pos = None
            self.update()
        elif event.type() == QEvent.MouseMove:
            if self._draggingArea:
                self._drag_last = event.pos()
            if self._draggingView:
                offset = QPointF(self._drag_last - event.pos())
                self._drag_last = event.pos()
                self.offset.setX(self.offset.x() + offset.x() / (self._last_width * self.scale_factor_x))
                self.offset.setY(self.offset.y() - offset.y() / (self._last_height * self.scale_factor_y))
                self._transform = None
                self.update()

            if self._transform is None:
                self.make_transform()
            if self._transform_inv is not None:
                self._cursor_pos = QPointF(event.pos())
                self.update()
            event.accept()
            return True
        elif event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            if self._transform is None:
                self.make_transform()
            pre_transform_inv = self._transform_inv

            delta = event.angleDelta()
            relative_scale = 1.25 ** (delta.y() / 120)
            self.scale_factor_x *= relative_scale
            self.scale_factor_y *= relative_scale

            self.make_transform()

            if pre_transform_inv is not None:
                pre_pos = event.posF()
                pre_pos_trans = pre_transform_inv.map(pre_pos)
                post_pos = self._transform.map(pre_pos_trans)

                self.offset.setX(self.offset.x() - (pre_pos.x() - post_pos.x()) / (self._last_width * self.scale_factor_x))
                self.offset.setY(self.offset.y() + (pre_pos.y() - post_pos.y()) / (self._last_height * self.scale_factor_y))

            self._transform = None
            self.update()
            event.accept()
            return True
        elif event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_R:
                self.reset_transform()
                event.accept()
                return True
        return False

    def reset_transform(self):
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        self.offset.setX(0.5)
        self.offset.setY(0.5)
        self._transform = None
        self.update()

    def preferred_size(self):
        return self._preferred_size

    def set_preferred_size(self, size):
        self._preferred_size = size
        self.updateGeometry()

    def set_smoothing(self, smoothing):
        self._smoothing = smoothing
        self._smoothing_kernel = None
        if self._smoothing is not None and self._smoothing > 0:
            self.make_smoothing_kernel()
        for data in self._data:
            data.update(self._smoothing_kernel, self._min, self._range)
        self.update()

    @except_print
    def _data_updated(self, new_index, loader_id):
        if self._initial_loading and not self._interactive_preload:
            return
        #if self._initial_loading and new_index % 10 != 0:
        #    return
        if self._smoothing_kernel is None and self._smoothing is not None and self._smoothing > 0:
            self.make_smoothing_kernel()
        if self._scalar_data is not None:
            found_data = None
            for data in self._data:
                if data.loader_id == loader_id:
                    found_data = data
                    break
            if found_data is None:
                found_data = ScalarDisplayData(self._scalar_data, loader_id)
                self._data.append(found_data)
            if found_data.update_range(self._min, self._range):
                for data in self._data:
                    if data.loader_id != loader_id:
                        data.update_range(self._min, self._range)
                for data in self._data:
                    data.update(self._smoothing_kernel, self._min, self._range)
            else:
                found_data.update(self._smoothing_kernel, self._min, self._range)
            self.update()

    def set_initial_loading(self, is_initial_loading):
        self._initial_loading = is_initial_loading
        if not self._interactive_preload:
            self.set_global_data(self._scalar_data, self._step)

    def set_interactive_preload(self, interactive_preload):
        self._interactive_preload = interactive_preload

    def set_global_data(self, data, step):
        """
        Args:
            data (data.scalar_data.ScalarEntry):
            step (int):
        """
        if self._smoothing_kernel is None and self._smoothing is not None and self._smoothing > 0:
            self.make_smoothing_kernel()
        if data != self._scalar_data:
            if self._scalar_data is not None:
                self._scalar_data.signals.step_added.disconnect(self._data_updated)
                self._scalar_data = None
                self._data = []

            if data is not None:
                assert not data.is_per_step()
                self._scalar_data = data
                for loader_id in data.loader_ids():
                    data = ScalarDisplayData(self._scalar_data, loader_id)
                    data.update_range(self._min, self._range)
                    self._data.append(data)
                for data in self._data:
                    data.update(self._smoothing_kernel, self._min, self._range)
                self._scalar_data.signals.step_added.connect(self._data_updated)
        self._step = step
        self.update()

    def reset(self):
        self._data = []
        self._scalar_data = None
        self.update()

    def save_image(self):
        save_file = QFileDialog.getSaveFileName(self, "Save image as", "", "Portable Network Graphics (*.png);;All Files (*)")
        if save_file and save_file[0]:
            self.save_file(save_file[0])

    def save_file(self, path):
        self.pixmap.save(path)
