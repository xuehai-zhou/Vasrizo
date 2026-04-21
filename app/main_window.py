"""Top-level window: menu, tab bar, sample browser, status bar.

Supports opening a dataset via:
  - File > Open Dataset Folder…  (Ctrl+O): folder picker with auto-detect
  - File > Configure Dataset…    (Ctrl+Shift+O): manual images/labels/output dialog
  - Drag and drop: drop a folder onto the window
"""
from __future__ import annotations
import os
from typing import Optional

from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QAction, QKeySequence, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QDockWidget, QMessageBox, QLabel,
    QFileDialog, QWidget, QVBoxLayout,
)

from .annotation_tab import AnnotationTab
from .io.data_loader import list_samples
from .models.app_settings import AppSettings
from .ui.dataset_dialog import DatasetDialog
from .ui.sample_browser import SampleBrowser
from .ui.status_bar import AppStatusBar
from .utils.layout_detect import detect_layout, default_output_dir


# Widget shown as the central widget when no tabs are open
class _EmptyState(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.label = QLabel(
            "<div style='text-align:center;'>"
            "<h2 style='color:#555;'>No dataset loaded</h2>"
            "<p style='color:#888; font-size: 11pt;'>"
            "Drop a dataset folder here,<br>"
            "or use <b>File → Open Dataset Folder…</b> (Ctrl+O)"
            "</p></div>")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("Vasrizo")
        self.resize(1500, 950)
        self.setAcceptDrops(True)

        # Central tab widget with an empty-state placeholder
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._empty = _EmptyState()
        self._central_stack: list[QWidget] = [self._empty, self.tabs]
        self.setCentralWidget(self._empty)

        # Left-dock sample browser
        self.browser = SampleBrowser()
        self.browser.openSample.connect(self.open_sample)
        self._browser_dock = QDockWidget("Samples", self)
        self._browser_dock.setWidget(self.browser)
        self._browser_dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._browser_dock)

        # Status bar
        self.status = AppStatusBar(self)
        self.setStatusBar(self.status)

        # Menus
        self._build_menu()

        # Initial state
        if self.settings.is_configured():
            self._refresh_sample_list()
            self.status.say(f"Loaded dataset: {self.settings.images_dir}")
        else:
            self.status.say("No dataset loaded. File → Open Dataset Folder…")

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        act_open = QAction("Open Dataset Folder…", self)
        act_open.setShortcut(QKeySequence.Open)  # Ctrl+O
        act_open.triggered.connect(self._pick_and_open_dataset)
        file_menu.addAction(act_open)

        act_configure = QAction("Configure Dataset…", self)
        act_configure.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_configure.triggered.connect(self._configure_dataset)
        file_menu.addAction(act_configure)

        file_menu.addSeparator()

        act_open_next = QAction("Open Next Pending Sample", self)
        act_open_next.setShortcut(QKeySequence("Ctrl+N"))
        act_open_next.triggered.connect(self._open_next_pending)
        file_menu.addAction(act_open_next)

        act_close = QAction("Close Tab", self)
        act_close.setShortcut(QKeySequence("Ctrl+W"))
        act_close.triggered.connect(self._close_current_tab)
        file_menu.addAction(act_close)

        file_menu.addSeparator()

        act_save = QAction("Save Current Tab", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._save_current_tab)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_refresh = QAction("Refresh Sample List", self)
        act_refresh.setShortcut(QKeySequence("F5"))
        act_refresh.triggered.connect(self._refresh_sample_list)
        file_menu.addAction(act_refresh)

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = self.menuBar().addMenu("&View")
        act_reset_view = QAction("Reset 3D View", self)
        act_reset_view.setShortcut(QKeySequence("R"))
        act_reset_view.triggered.connect(self._reset_current_view)
        view_menu.addAction(act_reset_view)

        view_menu.addSeparator()
        act_toggle_label = QAction("Toggle Label Visibility", self)
        act_toggle_label.setShortcut(QKeySequence("L"))
        act_toggle_label.triggered.connect(self._toggle_label_on_current_tab)
        view_menu.addAction(act_toggle_label)

        act_toggle_ct = QAction("Toggle Thresholded CT Visibility", self)
        act_toggle_ct.setShortcut(QKeySequence("T"))
        act_toggle_ct.triggered.connect(self._toggle_ct_on_current_tab)
        view_menu.addAction(act_toggle_ct)

    # ------------------------------------------------------------------
    # Dataset opening
    # ------------------------------------------------------------------

    def _pick_and_open_dataset(self):
        """File > Open Dataset Folder… — pick a folder, then auto-detect."""
        start = os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(
            self, "Open Dataset Folder", start)
        if folder:
            self._open_dataset_folder(folder)

    def _configure_dataset(self):
        """File > Configure Dataset… — always show the manual dialog."""
        vals = DatasetDialog.run(
            self,
            images=self.settings.images_dir,
            labels=self.settings.labels_dir,
            output=self.settings.output_dir,
        )
        if vals:
            imgs, lbls, out = vals
            self._apply_dirs(imgs, lbls, out)

    def _open_dataset_folder(self, folder: str):
        """Auto-detect images/labels inside `folder`; fall back to dialog."""
        if not os.path.isdir(folder):
            QMessageBox.warning(
                self, "Not a folder",
                f"Not a directory:\n{folder}")
            return

        images, labels = detect_layout(folder)
        output = default_output_dir(labels) if labels else ""

        if images and labels:
            self.status.say(
                f"Auto-detected: images={os.path.basename(images)}, "
                f"labels={os.path.basename(labels)}")
            self._apply_dirs(images, labels, output)
            return

        # Partial or no detection — open the dialog with whatever we have
        self.status.say(
            f"Couldn't auto-detect layout in {folder}. "
            "Please configure manually.")
        vals = DatasetDialog.run(
            self,
            images=images or folder,
            labels=labels or folder,
            output=output or default_output_dir(labels or folder),
        )
        if vals:
            imgs, lbls, out = vals
            self._apply_dirs(imgs, lbls, out)

    def _apply_dirs(self, images: str, labels: str, output: str):
        self.settings.set_dirs(images_dir=images, labels_dir=labels,
                               output_dir=output)
        # Swap central widget from empty state → tabs if not already
        if self.centralWidget() is self._empty:
            self.setCentralWidget(self.tabs)
        self._refresh_sample_list()
        self.status.say(
            f"Dataset loaded — images: {images}, labels: {labels}")

    # ------------------------------------------------------------------
    # Drag-and-drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._first_folder_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragEnterEvent):
        # Needed on some platforms to keep accepting the drag
        if self._first_folder_from_mime(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        folder = self._first_folder_from_mime(event.mimeData())
        if folder:
            event.acceptProposedAction()
            self._open_dataset_folder(folder)
        else:
            event.ignore()

    @staticmethod
    def _first_folder_from_mime(mime: QMimeData) -> Optional[str]:
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if os.path.isdir(path):
                return path
        return None

    # ------------------------------------------------------------------
    # Sample list
    # ------------------------------------------------------------------

    def _refresh_sample_list(self):
        if not self.settings.is_configured():
            self.browser.set_samples([], completed=set())
            self.status.say("No dataset configured.")
            return
        names = list_samples(self.settings.labels_dir, self.settings.images_dir)
        completed = set()
        out = self.settings.output_dir
        if os.path.isdir(out):
            completed = {f.replace(".nii.gz", "")
                         for f in os.listdir(out) if f.endswith(".nii.gz")}
        self.browser.set_samples(names, completed=completed)
        self.status.say(f"{len(names)} samples ({len(completed)} completed).")

    def _open_next_pending(self):
        if not self.settings.is_configured():
            QMessageBox.information(self, "No dataset",
                                    "Open a dataset folder first (Ctrl+O).")
            return
        names = list_samples(self.settings.labels_dir, self.settings.images_dir)
        completed = set()
        out = self.settings.output_dir
        if os.path.isdir(out):
            completed = {f.replace(".nii.gz", "")
                         for f in os.listdir(out) if f.endswith(".nii.gz")}
        open_names = {self.tabs.tabText(i).rstrip(" *")
                      for i in range(self.tabs.count())}
        for n in names:
            if n in completed or n in open_names:
                continue
            self.open_sample(n)
            return
        QMessageBox.information(self, "All done",
                                "No more pending samples.")

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def open_sample(self, name: str):
        if not self.settings.is_configured():
            QMessageBox.information(self, "No dataset",
                                    "Open a dataset folder first.")
            return
        # If already open, just focus
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).rstrip(" *") == name:
                self.tabs.setCurrentIndex(i)
                return
        # Make sure central widget is the tab bar
        if self.centralWidget() is self._empty:
            self.setCentralWidget(self.tabs)
        tab = AnnotationTab(name, self.settings, parent=self)
        tab.statusMessage.connect(self.status.say)
        tab.dirtyChanged.connect(lambda d, t=tab: self._update_tab_dirty(t, d))
        tab.sampleSaved.connect(self.browser.mark_completed)
        idx = self.tabs.addTab(tab, name)
        self.tabs.setCurrentIndex(idx)

    def _update_tab_dirty(self, tab: AnnotationTab, dirty: bool):
        idx = self.tabs.indexOf(tab)
        if idx < 0:
            return
        base = tab.sample_name
        self.tabs.setTabText(idx, f"{base} *" if dirty else base)
        self.status.set_dirty(dirty, base)

    def _on_tab_changed(self, idx: int):
        if idx < 0:
            self.status.set_dirty(False)
            return
        tab = self.tabs.widget(idx)
        if isinstance(tab, AnnotationTab):
            self.status.set_dirty(tab.is_dirty(), tab.sample_name)

    def _on_tab_close_requested(self, idx: int):
        tab = self.tabs.widget(idx)
        if isinstance(tab, AnnotationTab) and tab.is_dirty():
            ans = QMessageBox.question(
                self, "Unsaved changes",
                f"{tab.sample_name} has unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save)
            if ans == QMessageBox.Cancel:
                return
            if ans == QMessageBox.Save:
                tab._on_save()
        if isinstance(tab, AnnotationTab):
            tab.shutdown()
        self.tabs.removeTab(idx)
        tab.deleteLater()
        # Back to empty state if this was the last tab
        if self.tabs.count() == 0:
            self.setCentralWidget(self._empty)

    def _close_current_tab(self):
        idx = self.tabs.currentIndex()
        if idx >= 0:
            self._on_tab_close_requested(idx)

    def _save_current_tab(self):
        tab = self.tabs.currentWidget()
        if isinstance(tab, AnnotationTab):
            tab._on_save()

    def _reset_current_view(self):
        tab = self.tabs.currentWidget()
        if isinstance(tab, AnnotationTab):
            tab.viewer.reset_view()

    def _toggle_label_on_current_tab(self):
        tab = self.tabs.currentWidget()
        if isinstance(tab, AnnotationTab):
            tab.toggle_label_visibility()

    def _toggle_ct_on_current_tab(self):
        tab = self.tabs.currentWidget()
        if isinstance(tab, AnnotationTab):
            tab.toggle_ct_visibility()

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        dirty_tabs = [self.tabs.widget(i) for i in range(self.tabs.count())
                      if isinstance(self.tabs.widget(i), AnnotationTab)
                      and self.tabs.widget(i).is_dirty()]
        if dirty_tabs:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                f"{len(dirty_tabs)} tab(s) have unsaved changes. Quit anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                event.ignore()
                return
        # Best-effort shutdown of every tab's background threads. Using
        # try/except so one bad tab can't prevent cleanup of the others,
        # and so we never re-raise into Qt's teardown path.
        for i in range(self.tabs.count()):
            t = self.tabs.widget(i)
            if isinstance(t, AnnotationTab):
                try:
                    t.shutdown()
                except Exception as e:
                    print(f"[closeEvent] shutdown failed for tab {i}: {e}")
        super().closeEvent(event)
