from abc import abstractmethod

from PyQt5.QtCore import QRect, QSize, QPoint, Qt, QObject
from PyQt5.QtWidgets import QLayout, QSizePolicy

from tensorflow_viewer.utils import except_print


class OrderedItem(QObject):
    @property
    @abstractmethod
    def orderTag(self):
        ...


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=-1):
        super(FlowLayout, self).__init__(parent)

        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

        self.setSpacing(spacing)

        self.itemList = []
        self.sortedItemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)
        self.sortedItemList.append(item)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]

        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            popped = self.itemList.pop(index)
            self.sortedItemList.remove(popped)
            return popped

        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()

        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())

        margin, _, _, _ = self.getContentsMargins()

        size += QSize(2 * margin, 2 * margin)
        return size

    @except_print
    def resort(self):
        # Bubble sort should be efficient, because usually the list is sorted already
        for passnum in range(len(self.sortedItemList) - 1, 0, -1):
            had_change = False
            widget = self.sortedItemList[0].widget()
            if isinstance(widget, OrderedItem):
                last_key = widget.orderTag
            else:
                last_key = None
            for i in range(1, len(self.sortedItemList)):
                key = None
                widget = self.sortedItemList[i].widget()
                if isinstance(widget, OrderedItem):
                    key = widget.orderTag
                if (key is not None and last_key is None) or (key is not None and last_key is not None and key < last_key):
                    had_change = True
                    temp = self.sortedItemList[i]
                    self.sortedItemList[i] = self.sortedItemList[i-1]
                    self.sortedItemList[i-1] = temp
                last_key = key
            if not had_change:
                break

    @except_print
    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        lineItems = []
        lines = [lineItems]

        self.resort()

        for item in self.sortedItemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton,
                                                                Qt.Horizontal)
            spaceY = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton,
                                                                Qt.Vertical)
            if wid.isVisible():
                nextX = x + item.sizeHint().width() + spaceX
                if nextX - spaceX > rect.right() and lineHeight > 0:
                    x = rect.x()
                    y = y + lineHeight + spaceY
                    nextX = x + item.sizeHint().width() + spaceX
                    lineHeight = 0

                    lineItems = [(item, x, y, nextX - spaceX)]
                    lines.append(lineItems)
                else:
                    lineItems.append((item, x, y, nextX - spaceX))

                x = nextX
                lineHeight = max(lineHeight, item.sizeHint().height())
            else:
                lineItems.append((item, x, y, x))
        if not testOnly:
            for line in lines:
                if not line:
                    continue
                offset_x = (rect.width() - (line[-1][3] - line[0][1])) // 2
                for item in line:
                    item[0].setGeometry(QRect(QPoint(offset_x + item[1], item[2]), item[0].sizeHint()))

        return y + lineHeight - rect.y()

    """def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal)
            spaceY = self.spacing() + wid.style().layoutSpacing(QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical)
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()"""