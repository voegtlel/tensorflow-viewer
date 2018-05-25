import argparse
import os
import sys

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QDesktopWidget, QApplication, QAction, qApp, QMainWindow, QFileDialog, QVBoxLayout, QSlider, \
    QWidget, QScrollArea, QLabel, QCheckBox, QDoubleSpinBox, QHBoxLayout, QSplitter

from tensorflow_viewer.event_widget import EventWidget
from tensorflow_viewer.data.loader import DataLoader
from tensorflow_viewer.step_widget import StepWidget
from tensorflow_viewer.treeselect_widget import TreeSelectWidget
from tensorflow_viewer.utils import except_print

# Required imports, so they register with the loader
# noinspection PyUnresolvedReferences
import data.event_loader
# noinspection PyUnresolvedReferences
import data.events_loader
# noinspection PyUnresolvedReferences
import data.tfrecords_loader


class EventViewerWidget(QMainWindow):
    scale_sizes = [
        QSize(50, 50), QSize(100, 100), QSize(200, 200), QSize(300, 300), QSize(400, 400), QSize(600, 600),
        QSize(750, 750), QSize(1000, 1000), QSize(1500, 1500), QSize(2000, 2000)
    ]

    def __init__(self, initial_files=None, interactive_preload=False):
        # noinspection PyArgumentList
        super().__init__()

        #: :type: data.loader.DataLoader
        self.events_loader = None

        self.interactive_preload = interactive_preload

        self._last_files = None

        self.resize(800, 600)
        self.center()

        central_splitter = QSplitter(Qt.Horizontal, self)

        central_widget = QWidget(central_splitter)

        self.event_scroll_widget = QScrollArea(central_widget)
        self.event_scroll_widget.setWidgetResizable(True)
        self.event_widget = EventWidget(self.event_scroll_widget)
        self.event_scroll_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.event_scroll_widget.setWidget(self.event_widget)
        self.event_widget.set_interactive_preload(self.interactive_preload)

        self.step_widget = StepWidget(central_widget)
        self.step_widget.stepChanged.connect(self.step_change)

        vbox = QVBoxLayout(central_widget)
        vbox.addWidget(self.event_scroll_widget, 1)
        vbox.addWidget(self.step_widget, 0)
        central_widget.setLayout(vbox)

        central_splitter.addWidget(central_widget)
        central_splitter.setCollapsible(0, False)

        self.tree_select_widget = TreeSelectWidget(central_splitter)
        self.tree_select_widget.visible_changed.connect(self._tag_visible_changed)
        central_splitter.addWidget(self.tree_select_widget)
        central_splitter.setCollapsible(1, True)

        central_splitter.setSizes([1, 0])

        self.setCentralWidget(central_splitter)

        #: :type: PyQt5.QtWidgets.QToolbar
        toolbar = self.addToolBar("Toolbar")

        loadAct = QAction('L&oad', self)
        loadAct.setShortcut('Ctrl+O')
        loadAct.setStatusTip('Load an events file')
        loadAct.triggered.connect(self.load)
        toolbar.addAction(loadAct)

        loadDirAct = QAction('Load &Dir', self)
        loadDirAct.setShortcut('Ctrl+Shift+O')
        loadDirAct.setStatusTip('Load an events directory')
        loadDirAct.triggered.connect(self.load_dir)
        toolbar.addAction(loadDirAct)

        loadDirAddAct = QAction('Load &Dir to current View', self)
        loadDirAddAct.setShortcut('Ctrl+Shift+Alt+O')
        loadDirAddAct.setStatusTip('Load an events directory and adds it to the current view')
        loadDirAddAct.triggered.connect(self.load_dir_add)
        self.addAction(loadDirAddAct)

        saveAct = QAction('&Save All', self)
        saveAct.setShortcut('Ctrl+Alt+S')
        saveAct.setStatusTip('Saves all current images to files')
        saveAct.triggered.connect(self.save_all)
        toolbar.addAction(saveAct)

        toolbar.addSeparator()

        label = QLabel(self)
        label.setText("Scale:")
        toolbar.addWidget(label)
        self.scale_slider = QSlider(Qt.Horizontal, toolbar)
        self.scale_slider.setMinimum(0)
        self.scale_slider.setMaximum(10)
        self.scale_slider.setValue(2)
        self.scale_slider.setPageStep(1)
        self.scale_slider.valueChanged.connect(self.scale_change)
        toolbar.addWidget(self.scale_slider)

        toolbar.addSeparator()

        self.sync_step = QCheckBox("Sync Step", toolbar)
        self.sync_step.setChecked(True)
        self.sync_step.stateChanged.connect(self.sync_change)
        toolbar.addWidget(self.sync_step)

        self.smoothing = QDoubleSpinBox(toolbar)
        self.smoothing.setMinimum(0.0)
        self.smoothing.setMaximum(100.0)
        self.smoothing.setValue(1.0)
        self.smoothing.valueChanged.connect(self.smoothing_change)
        toolbar.addWidget(self.smoothing)

        toolbar.addSeparator()

        reset_views_action = QAction('&Reset Views', self)
        reset_views_action.setStatusTip('Resets all viewports')
        reset_views_action.setShortcut("Ctrl+R")
        reset_views_action.triggered.connect(self.reset_views)
        toolbar.addAction(reset_views_action)

        reload_file_action = QAction('Reload File', self)
        reload_file_action.setStatusTip('Reloads the current file')
        reload_file_action.setShortcut("F5")
        reload_file_action.triggered.connect(self.reload_file)
        self.addAction(reload_file_action)

        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        self.addAction(exitAct)

        self.statusBar()

        self.setWindowTitle('Events Viewer')
        self.show()

        if initial_files is not None:
            import glob
            files = []
            for initial_file in initial_files:
                files.extend(glob.glob(initial_file))
            self.load_files(files)

    def load_files(self, paths):
        self.load_stop()
        self.event_widget.set_initial_loading(True)
        self.events_loader = DataLoader(paths, interactive_preload=self.interactive_preload)
        self._last_files = paths
        self.setWindowTitle(', '.join(paths))
        self.events_loader.load_done.connect(self.load_done)
        self.events_loader.load_status.connect(self.load_status)
        self.events_loader.load_step.connect(self.load_step)
        self.events_loader.load_stop.connect(self.load_stop)
        self.events_loader.load_tag.connect(self.load_tag)
        self.events_loader.load_entry_global.connect(self.load_entry_global)
        self.events_loader.load_clear.connect(self.load_clear)
        self.events_loader.start()

    @except_print
    def load(self, _):
        if self._last_files:
            dir = os.path.commonpath(self._last_files)
        else:
            dir = ""
        selections = QFileDialog.getOpenFileNames(self, "Open Events File", dir, "Events Files (events.out.tfevents.*;summary.tfevents);;TFRecord Files (*.tfrecords);;All Files (*)")
        if selections and selections[0]:
            self.load_files(selections[0])

    @except_print
    def load_dir(self, _):
        if self._last_files:
            dir = os.path.commonpath(self._last_files)
        else:
            dir = ""
        selection = QFileDialog.getExistingDirectory(self, "Open Events File", dir)
        if selection is not None and selection:
            self.load_files([selection])

    @except_print
    def load_dir_add(self, _):
        if self._last_files:
            dir = os.path.commonpath(self._last_files)
        else:
            dir = ""
        selection = QFileDialog.getExistingDirectory(self, "Open Events File", dir)
        if selection is not None and selection:
            if self.events_loader is None:
                self.load_files([selection])
            else:
                self.event_widget.set_initial_loading(True)
                self.events_loader.add_load(selection)
                self.setWindowTitle(self.windowTitle() + ', ' + selection)

    def save_files(self, parent_path):
        self.event_widget.save_files(parent_path)

    @except_print
    def save_all(self):
        if self._last_files:
            dir = os.path.commonpath(self._last_files)
        else:
            dir = ""
        directory = QFileDialog.getExistingDirectory(self, "Save Images As", dir)
        if directory:
            self.save_files(directory)

    @except_print
    def load_stop(self):
        self.statusBar().showMessage("Closed events file")
        if self.events_loader is not None:
            self.events_loader.load_done.disconnect(self.load_done)
            self.events_loader.load_status.disconnect(self.load_status)
            self.events_loader.load_step.disconnect(self.load_step)
            self.events_loader.load_stop.disconnect(self.load_stop)
            self.events_loader.load_tag.disconnect(self.load_tag)
            self.events_loader.load_entry_global.disconnect(self.load_entry_global)
            self.events_loader.load_clear.disconnect(self.load_clear)
            self.events_loader.stop()
            self.events_loader = None
        self.step_widget.setSteps([])
        self.event_widget.clear()
        self.event_widget.set_lock_step_end(False)
        self.tree_select_widget.clear()
        self.step_widget.set_lock_step_end(False)

    @except_print
    def load_done(self):
        self.statusBar().showMessage("Events loaded")
        self.event_widget.set_lock_step_end(True)
        self.step_widget.set_lock_step_end(True)
        self.event_widget.set_initial_loading(False)
        if not self.interactive_preload and self.events_loader is not None:
            self.step_widget.setSteps(list(self.events_loader.steps))
            self.event_widget.set_events(self.events_loader.tag_index)

    @except_print
    def load_status(self, status, progress):
        self.statusBar().showMessage("Loaded {} events ({:.2%})".format(status, progress))

    @except_print
    def load_step(self, insert_index, step):
        if self.events_loader is not None:
            self.step_widget.setSteps(list(self.events_loader.steps))
            self.event_widget.set_events(self.events_loader.tag_index)

    @except_print
    def load_tag(self, tag, tag_type):
        self.event_widget.add_tag(tag, tag_type)
        self.tree_select_widget.add_tag(tag, tag_type)

    @except_print
    def load_entry_global(self, tag, tag_type):
        self.event_widget.add_global_entry(self.events_loader.get_global_entry(tag))
        self.tree_select_widget.add_global_entry(self.events_loader.get_global_entry(tag))

    @except_print
    def load_clear(self):
        self.load_files(self._last_files)

    @except_print
    def step_change(self, step, _):
        self.event_widget.set_step(step)

    @except_print
    def scale_change(self):
        slider_position = self.scale_slider.value()
        self.event_widget.set_viewer_preferred_size(self.scale_sizes[slider_position])

    @except_print
    def sync_change(self, do_sync):
        self.step_widget.setVisible(do_sync)
        self.event_widget.set_allow_step_select(not do_sync)

    @except_print
    def smoothing_change(self, value):
        self.event_widget.set_smoothing(value)

    @except_print
    def reset_views(self):
        self.event_widget.reset_views()

    @except_print
    def reload_file(self):
        if self._last_files is not None:
            self.load_files(self._last_files)

    @except_print
    def _tag_visible_changed(self, tags, visible):
        self.event_widget.set_tags_visible(tags, visible)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initial_file', action='store', nargs='?', default=None)  # optional positional argument
    parser.add_argument('--interactive-preload', action='store_true', default=None)  # optional positional argument

    parsed_args, unparsed_args = parser.parse_known_args()

    app = QApplication(sys.argv)
    ex = EventViewerWidget([parsed_args.initial_file] if parsed_args.initial_file else None, interactive_preload=parsed_args.interactive_preload)
    sys.exit(app.exec_())
