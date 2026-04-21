#!/usr/bin/env python3
"""Vasrizo — desktop annotation app entry point.

Launch plain (`python main.py`) and use File > Open Dataset Folder… or
drag a folder onto the window. CLI args below are optional shortcuts
for development / scripted launches.
"""
from __future__ import annotations
import argparse
import os
import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.models.app_settings import AppSettings
from app.utils.config import DataPaths
from app.utils.layout_detect import detect_layout, default_output_dir


def parse_args():
    p = argparse.ArgumentParser(description="Vasrizo annotation tool")
    # Either pass --folder (auto-detect) or the explicit trio below.
    p.add_argument("--folder", default=None,
                   help="Dataset folder (auto-detect images/ and labels/ inside)")
    p.add_argument("--images_dir", default=None,
                   help="Absolute path to images directory")
    p.add_argument("--labels_dir", default=None,
                   help="Absolute path to labels directory")
    p.add_argument("--output_dir", default=None,
                   help="Absolute path to output directory")
    p.add_argument("--sample", default=None,
                   help="After loading, open this sample in a tab")
    return p.parse_args()


def main():
    args = parse_args()

    settings = AppSettings(paths=DataPaths())  # defaults kept but unused unless nothing else is given

    # Resolve CLI args: explicit trio > --folder > DataPaths defaults
    if args.images_dir and args.labels_dir:
        settings.set_dirs(
            images_dir=args.images_dir,
            labels_dir=args.labels_dir,
            output_dir=args.output_dir or default_output_dir(args.labels_dir),
        )
    elif args.folder:
        imgs, lbls = detect_layout(args.folder)
        if imgs and lbls:
            settings.set_dirs(
                images_dir=imgs, labels_dir=lbls,
                output_dir=args.output_dir or default_output_dir(lbls))
        else:
            print(f"[main] Could not auto-detect layout in {args.folder}. "
                  "Use File > Configure Dataset… to set dirs manually.")

    app = QApplication(sys.argv)
    app.setApplicationName("Vasrizo")

    win = MainWindow(settings)
    win.show()

    if args.sample and settings.is_configured():
        win.open_sample(args.sample)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
