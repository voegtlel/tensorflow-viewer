import bisect

from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, pyqtSignal
from PyQt5.QtWidgets import QTreeView

from tensorflow_viewer.utils import except_print


def extract_tag_name(tag_name):
    assert isinstance(tag_name, str)
    if '/' in tag_name:
        idx = tag_name.index('/')
        tag_name_pure = tag_name[:idx]
        next_tag_name = tag_name[idx + 1:]
        if '/' in next_tag_name:
            idx = next_tag_name.index('/')
            next_tag_name_pure = next_tag_name[:idx]
        else:
            next_tag_name_pure = next_tag_name
    else:
        tag_name_pure = tag_name
        next_tag_name = None
        next_tag_name_pure = None
    return tag_name_pure, next_tag_name, next_tag_name_pure


class StructuredTagRoot:
    def __init__(self):
        self.name = "root"
        self.parent = None
        #: :type: list[str]
        self.tag_values = []
        #: :type: dict[str, StructuredTag]
        self.sub_tags = dict()
        self.paths = None
        self.tag_type = None
        self._visible = True

    def add_tag(self, tag_path, tag_type, begin_add=None, end_add=None, tag_name=None):
        if tag_name is None:
            tag_name = 'root/' + tag_path[0]

        tag_name_pure, next_tag_name, next_tag_name_pure = extract_tag_name(tag_name)

        assert tag_name_pure == self.name, "Invalid path {} for {}".format(tag_path, self)
        if next_tag_name is None:
            assert self.paths is not None
            assert self.tag_type == tag_type, "Invalid tag type {} for {}".format(tag_type, self)
            self.paths.append(tag_path)
        else:
            assert self.sub_tags is not None, tag_path
            sub_tag = self.sub_tags.get(next_tag_name_pure)
            if sub_tag is None:
                idx = bisect.bisect_right(self.tag_values, next_tag_name_pure)
                if begin_add is not None:
                    begin_add(self, idx)
                self.tag_values.insert(idx, next_tag_name_pure)
                self.sub_tags[next_tag_name_pure] = StructuredTag(tag_path, tag_type, self, next_tag_name)
                if end_add is not None:
                    end_add()
            else:
                sub_tag.add_tag(tag_path, tag_type, begin_add, end_add, next_tag_name)

    @property
    def fully_visible(self):
        if not self._visible:
            return False
        if self.sub_tags is None:
            return True
        return all(sub_tag.fully_visible for sub_tag in self.sub_tags.values())

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, visible):
        self._visible = visible
        if visible:
            parent = self.parent
            while parent is not None:
                if parent._visible:
                    break
                parent._visible = True
                parent = parent.parent
        if self.sub_tags is not None:
            for sub_tag in self.sub_tags.values():
                sub_tag.visible = visible

    def last_tag(self):
        if self.sub_tags is None:
            return self
        last_key = self.tag_values[len(self.tag_values) - 1]
        return self.sub_tags[last_key].last_tag()

    def __str__(self):
        if self.sub_tags is None:
            return "name=" + repr(self.name)
        return "name=" + repr(self.name) + ", subitems={" + ", ".join(repr(key) + ': ' + repr(self.sub_tags[key]) for key in self.tag_values) + "}"

    def __repr__(self):
        return self.__str__()

    def all_paths(self):
        if self.paths is not None:
            for path in self.paths:
                yield path
        else:
            for key in self.tag_values:
                sub_tag = self.sub_tags[key]
                for path in sub_tag.all_paths():
                    yield path


class StructuredTag(StructuredTagRoot):
    def __init__(self, tag_path, tag_type, parent=None, tag_name=None):
        super().__init__()

        self.name, next_tag_name, next_tag_name_pure = extract_tag_name(tag_name)
        if parent is not None:
            self._visible = parent.visible
        self.parent = parent
        if next_tag_name is None:
            self.tag_values = None
            self.sub_tags = None
            self.paths = [tag_path]
            self.tag_type = tag_type
        else:
            #: :type: list[str]
            self.tag_values = [next_tag_name_pure]
            #: :type: dict[str, StructuredTag]
            self.sub_tags = {next_tag_name_pure: StructuredTag(tag_path, tag_type, self, next_tag_name)}
            self.paths = None
            self.tag_type = None


class TreeModel(QAbstractItemModel):
    visible_changed = pyqtSignal(list, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = StructuredTagRoot()

    @except_print
    def index(self, row, column, parent):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        if parent.isValid():
            #: :type: StructuredTag
            parent_item = parent.internalPointer()
            if parent_item is not None and 0 <= row < len(parent_item.tag_values):
                tag_name = parent_item.tag_values[row]
                if tag_name in parent_item.sub_tags:
                    tag = parent_item.sub_tags[tag_name]
                    return self.createIndex(row, column, tag)
        else:
            if 0 <= row < len(self._root.tag_values):
                key = self._root.tag_values[row]
                tag = self._root.sub_tags[key]
                return self.createIndex(row, column, tag)
        return QModelIndex()

    def _item_index(self, item=None):
        assert item.parent is not None, "Index of root"
        return item.parent.tag_values.index(item.name)

    def _get_index(self, item):
        if item is self._root:
            return QModelIndex()
        if item is not None:
            return self.createIndex(self._item_index(item), 0, item)
        return QModelIndex()

    @except_print
    def parent(self, index):
        if index.isValid():
            #: :type: StructuredTag
            item = index.internalPointer()
            if item is not None:
                return self._get_index(item.parent)
        return QModelIndex()

    @except_print
    def rowCount(self, index):
        if index.isValid():
            #: :type: StructuredTag
            item = index.internalPointer()
            if item is not None:
                return len(item.tag_values)
        return len(self._root.tag_values)

    @except_print
    def hasChildren(self, parent):
        if parent.isValid():
            #: :type: StructuredTag
            item = parent.internalPointer()
            if item is not None:
                return item.tag_values is not None
        return True

    @except_print
    def columnCount(self, parent):
        return 1

    @except_print
    def data(self, index, role):
        if role == Qt.DisplayRole and index.isValid():
            #: :type: StructuredTag
            item = index.internalPointer()
            if item is not None:
                return item.name
        elif role == Qt.CheckStateRole and index.isValid() and index.column() == 0:
            #: :type: StructuredTag
            item = index.internalPointer()
            if item is not None:
                if item.fully_visible:
                    return Qt.Checked
                elif item.visible:
                    return Qt.PartiallyChecked
                return Qt.Unchecked
        return None

    def add_tag(self, tag_path, tag_type):
        def begin_add(item, idx):
            self.beginInsertRows(self._get_index(item), idx, idx)

        def end_add():
            self.endInsertRows()

        self._root.add_tag(tag_path, tag_type, begin_add, end_add)

    @except_print
    def headerData(self, section, orientation, role=None):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return "Name"

    @except_print
    def flags(self, index):
        if not index.isValid():
            return 0
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            flags |= Qt.ItemIsUserCheckable

        return flags

    @except_print
    def setData(self, index, value, role):
        if index.isValid() and index.column() == 0 and role == Qt.CheckStateRole:
            #: :type: StructuredTag
            item = index.internalPointer()
            item.visible = value
            last_item = item.last_tag()
            last_index = self._get_index(last_item)
            self.dataChanged.emit(index, last_index, [role])
            parent = item.parent
            while parent is not None:
                index = self._get_index(parent)
                self.dataChanged.emit(index, index, [role])
                parent = parent.parent
            self.visible_changed.emit(list(item.all_paths()), value)
            return True
        return False


class TreeSelectWidget(QTreeView):
    visible_changed = pyqtSignal(list, bool)

    def __init__(self, parent=None):
        super(TreeSelectWidget, self).__init__(parent)

        self._model = TreeModel(self)
        self._model.visible_changed.connect(self._visible_changed)
        self.setModel(self._model)

    def clear(self):
        self._model.visible_changed.disconnect(self._visible_changed)
        self._model = TreeModel(self)
        self._model.visible_changed.connect(self._visible_changed)
        self.setModel(self._model)

    def add_tag(self, tag, tag_type):
        """
        Args:
            tag (tuple[str|int]):
            tag_type (str):
        """
        self._model.add_tag(tag, tag_type)

    def add_global_entry(self, entry):
        """
        Args:
            entry (Entry):
        """
        self._model.add_tag(entry.tag, entry.type())

    @except_print
    def _visible_changed(self, tags, visible):
        self.visible_changed.emit(tags, visible)
