from typing import Union

from PyQt5.QtCore import QByteArray, QBuffer, QPointF, QSize, QEvent, QObject, Qt
from PyQt5.QtGui import QImageReader, QPixmap, QPaintEvent, QPainter, QTransform, QMouseEvent, \
    QWheelEvent, QPalette, QKeyEvent, QImage
from PyQt5.QtWidgets import QWidget, QAction, QMenu, QFileDialog

from tensorflow_viewer.utils import except_print


class ImageViewerWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageViewerWidget, self).__init__(parent)

        self.scale_factor = 1.0

        self.offset = QPointF(0, 0)

        self._preferred_size = QSize(200, 200)

        #: :type: QPixmap
        self.pixmap = None
        self._data = None
        self._image = None
        #: :type: data.image_data.ImageDataFuture
        self._image_data_future = None

        self._dragging = False
        self._drag_last = None

        self.setBackgroundRole(QPalette.Dark)
        self.installEventFilter(self)
        self.setFocusPolicy(Qt.WheelFocus)

    @except_print(no_wrap=True)
    def sizeHint(self):
        return self._preferred_size

    def make_transform(self, offset_override=None, scale_override=None):
        if offset_override is None:
            offset_override = self.offset
        if scale_override is None:
            scale_override = self.scale_factor
        transform = QTransform()
        target_width = self.width()
        target_height = self.height()
        # Center in widget
        transform.translate(offset_override.x(), offset_override.y())
        transform.translate(target_width / 2, target_height / 2)
        transform.scale(scale_override, scale_override)
        return transform

    @except_print(no_wrap=True)
    def paintEvent(self, paint_event: QPaintEvent):
        painter = QPainter()
        painter.begin(self)
        painter.eraseRect(0, 0, self.width(), self.height())
        if self.pixmap:
            # Transform by parent transformation
            painter.setTransform(self.make_transform())
            # Fit image to widget
            target_width = self.width()
            target_height = self.height()
            src_width = self.pixmap.width()
            src_height = self.pixmap.height()
            if target_width / target_height >= src_width / src_height:
                dst_height = target_height
                dst_width = src_width * target_height / src_height
            else:
                dst_width = target_width
                dst_height = src_height * target_width / src_width
            painter.scale(dst_width / src_width, dst_height / src_height)
            painter.translate(-src_width/2, -src_height/2)
            # Draw image centered
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
            painter.drawPixmap(0, 0, self.pixmap)
        painter.end()

    @except_print(no_wrap=True)
    def eventFilter(self, object: QObject, event: Union[QEvent, QMouseEvent, QWheelEvent, QKeyEvent]):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self._dragging = True
                self._drag_last = event.pos()
                event.accept()
                self.update()
                return True
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self._dragging = False
                event.accept()
                self.update()
                return True
            elif event.button() == Qt.RightButton:
                context_menu = QMenu(self)
                save_image_action = QAction("&Save Image")
                save_image_action.triggered.connect(self.save_image)
                context_menu.addAction(save_image_action)
                context_menu.exec_(event.globalPos())

        elif event.type() == QEvent.MouseMove:
            if self._dragging:
                offset = QPointF(self._drag_last - event.pos())
                self._drag_last = event.pos()
                self.offset -= offset
                event.accept()
                self.update()
                return True
        elif event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            pre_transform = self.make_transform()

            delta = event.angleDelta()
            relative_scale = 1.25 ** (delta.y() / 120)
            self.scale_factor *= relative_scale

            post_transform = self.make_transform()

            pre_transform_inv, succ = pre_transform.inverted()
            if succ:
                pre_pos = event.posF()
                pre_pos_trans = pre_transform_inv.map(pre_pos)
                post_pos = post_transform.map(pre_pos_trans)

                self.offset += (pre_pos - post_pos)
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
        self.scale_factor = 1.0
        self.offset.setX(0.0)
        self.offset.setY(0.0)
        self.update()

    def preferred_size(self):
        return self._preferred_size

    def set_preferred_size(self, size):
        self._preferred_size = size
        self.updateGeometry()

    def set_initial_loading(self, initial_loading):
        pass

    def set_interactive_preload(self, interactive_preload):
        pass

    def set_step_data(self, step_data):
        """
        Args:
            step_data (data.image_data.ImageEntry):
        """
        if self._image_data_future is not None:
            self._image_data_future.signals.data_ready.disconnect(self._set_image)
            self._image_data_future.signals.raw_data_ready.disconnect(self._set_image_raw)
            self._image_data_future.signals.done.disconnect(self._on_done_image_load)
            self._image_data_future.cancel()
            self._image_data_future = None

        if step_data is not None:
            self._image_data_future = step_data.read_image_data_thread()
            self._image_data_future.signals.data_ready.connect(self._set_image)
            self._image_data_future.signals.raw_data_ready.connect(self._set_image_raw)
            self._image_data_future.signals.done.connect(self._on_done_image_load)
            self._image_data_future.start()

    @except_print
    def _on_done_image_load(self):
        self._image_data_future = None

    @except_print
    def _set_image(self, data, description):
        if data is None:
            self.pixmap = None
            self._image = None
            self._data = None
        else:
            self._data = None
            byte_array = QByteArray(data)
            buffer = QBuffer(byte_array)
            buffer.open(QBuffer.ReadOnly)
            reader = QImageReader(buffer)
            image = reader.read()
            self._image = image
            if image:
                self.pixmap = QPixmap.fromImage(image)
            else:
                self.pixmap = None
        self.setToolTip(description)
        self.update()

    @except_print
    def _set_image_raw(self, data, width, height, is_rgb, description):
        if data is None:
            self.pixmap = None
            self._image = None
            self._data = None
        else:
            stride = width * (3 if is_rgb else 1)
            if stride % 4 != 0:
                stride = (stride // 4 + 1) * 4
            assert len(data) == stride * height
            self._data = data
            image = QImage(data, width, height, QImage.Format_RGB888 if is_rgb else QImage.Format_Grayscale8)
            self._image = image
            if image:
                self.pixmap = QPixmap.fromImage(image)
            else:
                self.pixmap = None
        self.setToolTip(description)
        self.update()

    def set_smoothing(self, smoothing):
        pass

    @except_print
    def save_image(self):
        save_file = QFileDialog.getSaveFileName(self, "Save image as", "", "Portable Network Graphics (*.png);;All Files (*)")
        if save_file and save_file[0]:
            self.save_file(save_file[0])

    def reset(self):
        self._set_image(None, None)

    def save_file(self, path):
        if self.pixmap is not None:
            self.pixmap.save(path)
