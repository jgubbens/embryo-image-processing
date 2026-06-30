"""
Napari embryo nuclear-cycle (NC) stage labeling tool.

Point it at:
  - a single image file, OR
  - a flat directory of images, OR
  - a directory of CHANNEL subdirectories (e.g. "transmitted/",
    "638_widefield/"), each holding the same set of positions imaged in
    a different channel. Matching images across channel folders (same
    filename except for the channel-specific part) are grouped and
    opened together as separate layers in napari, labeled once, and
    moved together.

    e.g.
        embryo_images/
          transmitted/0629_pos007_transmitted.tif
          638_widefield/0629_pos007_638_widefield.tif
    both belong to group key "0629_pos007" and open as two layers.

One button per developmental stage (undetectable -> NC14+). Clicking a
button records the *currently displayed frame* as the first frame of
that stage.

"Confirm & Save" immediately:
  1. writes/updates this group's entry in labels.yaml (preserving all
     existing comments/formatting/other entries) -- written ONCE per
     group, even though multiple channel images share it,
  2. moves every channel's image for this group into a "labeled"
     subfolder (one per channel folder, so channel structure is kept),
  3. loads the next group automatically.

Because each confirm saves right away, quitting the app at any point
keeps everything you've already confirmed -- nothing is lost.

Install requirements (once):
    pip install napari[all] magicgui ruamel.yaml

Usage:
    # Single image
    python embryo_labeler.py /path/to/embryo17.tif --yaml /path/to/labels.yaml

    # Flat directory of images -- one image at a time
    python embryo_labeler.py /path/to/embryo_images/ --yaml labels.yaml

    # Directory of channel subfolders -- matching images grouped & opened together
    python embryo_labeler.py /path/to/embryo_images/ --yaml labels.yaml

    # Optional flags
    python embryo_labeler.py /path/to/embryo_images/ \
        --yaml labels.yaml \
        --labeled-dir /path/to/labeled \
        --time-between-frames 60 \
        --extensions .tif,.tiff \
        --include-labeled
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import napari
from magicgui.widgets import Container, Label, PushButton, SpinBox
from qtpy.QtWidgets import QMessageBox, QScrollArea, QSizePolicy
from ruamel.yaml import YAML

# Stages in chronological order. This list both defines the buttons
# shown and the key order written back to the yaml file.
STAGES = [
    "undetectable",
    "NC9",
    "NC9M",
    "NC10",
    "NC10M",
    "NC11",
    "NC11M",
    "NC12",
    "NC12M",
    "NC13",
    "NC13M",
    "NC14+",
]

DEFAULT_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".nd2", ".czi", ".lsm"}

# Used internally as the "channel" key when there's no real channel split.
_SINGLE_CHANNEL = "_image"


class ImageGroup(NamedTuple):
    key: str  # embryo name / yaml key
    files: dict[str, Path]  # channel name -> file path


class EmbryoLabelerSession:
    def __init__(
        self,
        path: str | Path,
        yaml_path: str | Path,
        labeled_dir: str | Path | None = None,
        time_between_frames: int = 60,
        extensions: set[str] | None = None,
        include_labeled: bool = False,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        self.extensions = extensions or DEFAULT_EXTENSIONS
        self.include_labeled = include_labeled
        self.default_time_between_frames = time_between_frames
        # If set, overrides the per-channel "<channel folder>/labeled" default.
        self.labeled_dir_override = Path(labeled_dir) if labeled_dir else None

        self.yaml_path = Path(yaml_path)

        # ruamel round-trip yaml preserves comments and ordering
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

        if self.yaml_path.exists():
            with open(self.yaml_path) as f:
                self.data = self.yaml.load(f)
            if self.data is None:
                self.data = {}
        else:
            self.data = {}

        if path.is_dir():
            self.source_dir = path
            channel_dirs = self._find_channel_dirs(path)
            if channel_dirs:
                self.channel_mode = True
                self.queue = self._build_channel_queue(channel_dirs)
            else:
                self.channel_mode = False
                self.queue = self._build_flat_queue(path)
        else:
            self.source_dir = path.parent
            self.channel_mode = False
            self.queue = [ImageGroup(key=path.stem, files={_SINGLE_CHANNEL: path})]

        if not self.queue:
            raise SystemExit(
                f"No images left to label in {path} "
                f"(extensions: {sorted(self.extensions)})."
            )

        # current-group state, set in _load_next_image()
        self.embryo_name: str | None = None
        self.current_files: dict[str, Path] = {}
        self.labels: dict[str, int] = {}

        self.viewer = napari.Viewer()
        self._build_ui()
        self._load_next_image()

    # ------------------------------------------------------------------
    # Discovery / grouping
    # ------------------------------------------------------------------
    def _has_images(self, directory: Path) -> bool:
        return any(
            p.is_file() and p.suffix.lower() in self.extensions
            for p in directory.iterdir()
        )

    def _find_channel_dirs(self, directory: Path) -> list[Path]:
        subdirs = sorted(
            d for d in directory.iterdir() if d.is_dir() and d.name.lower() != "labeled"
        )
        return [d for d in subdirs if self._has_images(d)]

    def _build_flat_queue(self, directory: Path) -> list[ImageGroup]:
        files = sorted(
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in self.extensions
        )
        if not self.include_labeled:
            files = [p for p in files if p.stem not in self.data]
        return [ImageGroup(key=p.stem, files={_SINGLE_CHANNEL: p}) for p in files]

    @staticmethod
    def _channel_key(stem: str, channel_name: str) -> str:
        """Strip the channel folder's name out of a filename stem so that
        matching images across channel folders resolve to the same key.
        e.g. ("0629_pos007_transmitted", "transmitted") -> "0629_pos007"
             ("0629_pos007_638_widefield", "638_widefield") -> "0629_pos007"
        """
        key = stem
        if channel_name in key:
            key = key.replace(channel_name, "")
        key = key.strip("_- ")
        while "__" in key:
            key = key.replace("__", "_")
        return key

    def _build_channel_queue(self, channel_dirs: list[Path]) -> list[ImageGroup]:
        # channel name -> {group key -> file path}
        per_channel: dict[str, dict[str, Path]] = {}
        for d in channel_dirs:
            mapping: dict[str, Path] = {}
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in self.extensions:
                    key = self._channel_key(p.stem, d.name)
                    if key in mapping:
                        print(
                            f"[warning] duplicate key '{key}' in {d.name} "
                            f"({mapping[key].name} vs {p.name}); keeping the latter."
                        )
                    mapping[key] = p
            per_channel[d.name] = mapping

        key_sets = [set(m) for m in per_channel.values()]
        common_keys = set.intersection(*key_sets) if key_sets else set()
        all_keys = set.union(*key_sets) if key_sets else set()
        missing = all_keys - common_keys
        if missing:
            print(
                f"[warning] {len(missing)} position(s) missing in at least one "
                f"channel folder, skipping: {sorted(missing)}"
            )

        if not self.include_labeled:
            common_keys = {k for k in common_keys if k not in self.data}

        groups = []
        for key in sorted(common_keys):
            files = {channel: per_channel[channel][key] for channel in per_channel}
            groups.append(ImageGroup(key=key, files=files))
        return groups

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def _load_next_image(self):
        if not self.queue:
            QMessageBox.information(
                None, "All done", "No more images left to label in this folder."
            )
            self.viewer.close()
            return

        group = self.queue.pop(0)
        self.embryo_name = group.key
        self.current_files = group.files
        self.labels = {}

        self.viewer.layers.clear()
        for channel, file_path in self.current_files.items():
            layer_name = self.embryo_name if channel == _SINGLE_CHANNEL else channel
            opened = self.viewer.open(str(file_path), name=layer_name)
            if not opened:
                QMessageBox.warning(
                    None,
                    "Could not open",
                    f"napari could not open {file_path}, skipping this group.",
                )
                self._load_next_image()
                return

        # Preload existing entry if relabeling (--include-labeled)
        existing = self.data.get(self.embryo_name)
        tbf = self.default_time_between_frames
        if existing:
            for stage in STAGES:
                if stage in existing:
                    self.labels[stage] = int(existing[stage])
            tbf = int(existing.get("time_between_frames", tbf))

        self.time_between_frames.value = tbf
        for stage in STAGES:
            initial = f"f{self.labels[stage]}" if stage in self.labels else "—"
            self.status_labels[stage].value = initial

        remaining = len(self.queue)
        if self.channel_mode:
            channel_list = ", ".join(self.current_files.keys())
            subtitle = f"channels: {channel_list}"
        else:
            subtitle = self.current_files[_SINGLE_CHANNEL].name
        self.viewer.title = f"Labeling: {self.embryo_name} ({remaining} remaining)"
        self.name_label.value = f"Embryo: {self.embryo_name}\n{subtitle}\n{remaining} left after this"
        self._refresh_frame_indicator()

    # ------------------------------------------------------------------
    def current_frame(self) -> int:
        """1-indexed frame number currently shown (assumes axis 0 = time)."""
        step = self.viewer.dims.current_step
        if not step:
            return 1
        return int(step[0]) + 1

    def _refresh_frame_indicator(self, event=None):
        self.frame_indicator.value = f"Current frame: {self.current_frame()}"

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Build the dock panel once. Per-image state is reset by _load_next_image()."""
        widgets = []

        self.name_label = Label(value="Embryo: -")
        widgets.append(self.name_label)

        self.time_between_frames = SpinBox(
            value=self.default_time_between_frames,
            min=1,
            max=100000,
            label="time_between_frames (s)",
        )
        widgets.append(self.time_between_frames)

        self.frame_indicator = Label(value="Current frame: -")
        widgets.append(self.frame_indicator)
        self.viewer.dims.events.current_step.connect(self._refresh_frame_indicator)

        self.status_labels: dict[str, Label] = {}
        for stage in STAGES:
            btn = PushButton(text=stage)
            btn.native.setMinimumHeight(22)
            btn.native.setMaximumHeight(22)
            lbl = Label(value="—")
            lbl.native.setMinimumWidth(40)
            lbl.native.setMaximumWidth(50)
            self.status_labels[stage] = lbl

            def make_callback(s=stage, l=lbl):
                def callback():
                    frame = self.current_frame()
                    self.labels[s] = frame
                    l.value = f"f{frame}"

                return callback

            btn.changed.connect(make_callback())
            row = Container(widgets=[btn, lbl], layout="horizontal", labels=False)
            row.native.layout().setContentsMargins(0, 0, 0, 0)
            row.native.layout().setSpacing(4)
            widgets.append(row)

        clear_btn = PushButton(text="Clear all labels")

        def clear_all():
            self.labels.clear()
            for s in STAGES:
                self.status_labels[s].value = "—"

        clear_btn.changed.connect(clear_all)
        widgets.append(clear_btn)

        skip_btn = PushButton(text="Skip (no save)")
        skip_btn.changed.connect(self.skip)
        widgets.append(skip_btn)

        confirm_btn = PushButton(text="Confirm && Save")
        confirm_btn.changed.connect(self.confirm)
        widgets.append(confirm_btn)

        container = Container(widgets=widgets)
        container.native.layout().setContentsMargins(6, 6, 6, 6)
        container.native.layout().setSpacing(3)

        scroll = QScrollArea()
        scroll.setWidget(container.native)
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        scroll.setMinimumHeight(150)

        self.viewer.window.add_dock_widget(scroll, area="right", name="Embryo Labeler")

    # ------------------------------------------------------------------
    def skip(self):
        """Move to the next group without saving or moving the current one."""
        if self.labels:
            resp = QMessageBox.question(
                None,
                "Discard labels?",
                f"You've set {len(self.labels)} stage(s) for '{self.embryo_name}' "
                "that haven't been saved. Skip anyway and discard them?",
            )
            if resp != QMessageBox.Yes:
                return
        self._load_next_image()

    # ------------------------------------------------------------------
    def _dest_dir_for(self, channel: str, file_path: Path) -> Path:
        if self.labeled_dir_override is not None:
            if channel == _SINGLE_CHANNEL:
                return self.labeled_dir_override
            return self.labeled_dir_override / channel
        # default: a "labeled" subfolder right next to where the file came from
        return file_path.parent / "labeled"

    def confirm(self):
        if not self.labels:
            QMessageBox.warning(
                None, "Nothing to save", "Set at least one stage before confirming."
            )
            return

        # Sanity check: frame numbers should be non-decreasing in stage order
        ordered = [(s, self.labels[s]) for s in STAGES if s in self.labels]
        for (s1, f1), (s2, f2) in zip(ordered, ordered[1:]):
            if f2 < f1:
                resp = QMessageBox.question(
                    None,
                    "Out-of-order frames",
                    f"'{s2}' (frame {f2}) starts before '{s1}' (frame {f1}). "
                    "Save anyway?",
                )
                if resp != QMessageBox.Yes:
                    return
                break

        entry = {"time_between_frames": int(self.time_between_frames.value)}
        for stage in STAGES:
            if stage in self.labels:
                entry[stage] = self.labels[stage]

        # Save to yaml immediately (once per group) so nothing is lost if the
        # app quits later.
        is_new_entry = self.embryo_name not in self.data
        self.data[self.embryo_name] = entry
        if is_new_entry:
            # Match the existing file's style of a blank line between entries.
            self.data.yaml_set_comment_before_after_key(self.embryo_name, before="\n")
        self.yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.data, f)

        # Move every channel's file for this group immediately too.
        moved = []
        try:
            for channel, file_path in self.current_files.items():
                dest_dir = self._dest_dir_for(channel, file_path)
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / file_path.name
                shutil.move(str(file_path), str(dest))
                moved.append(dest)
        except (shutil.Error, OSError) as e:
            QMessageBox.critical(
                None,
                "Move failed",
                f"Label was saved to {self.yaml_path.name}, but moving files failed: {e}",
            )
            self._load_next_image()
            return

        self._load_next_image()


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Napari embryo NC-stage labeler")
    parser.add_argument(
        "path",
        help="Path to a single embryo movie/stack, a flat directory of them, "
        "or a directory of channel subfolders (e.g. transmitted/, 638_widefield/)",
    )
    parser.add_argument("--yaml", default="labels.yaml", help="Path to labels.yaml")
    parser.add_argument(
        "--labeled-dir",
        default=None,
        help="Base folder to move confirmed images into. For channel mode, each "
        "channel gets its own '<labeled-dir>/<channel>' subfolder. Default: a "
        "'labeled' subfolder next to each source file (per channel folder).",
    )
    parser.add_argument(
        "--time-between-frames",
        type=int,
        default=60,
        help="Default seconds between frames (editable per image in the UI)",
    )
    parser.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated list of file extensions to include when given a "
        f"directory, e.g. '.tif,.png' (default: {','.join(sorted(DEFAULT_EXTENSIONS))})",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="Also queue up images/groups whose key already has an entry in "
        "labels.yaml (default: skip them)",
    )
    args = parser.parse_args()

    extensions = None
    if args.extensions:
        extensions = {
            e if e.startswith(".") else f".{e}"
            for e in (x.strip().lower() for x in args.extensions.split(","))
            if e
        }

    EmbryoLabelerSession(
        path=args.path,
        yaml_path=args.yaml,
        labeled_dir=args.labeled_dir,
        time_between_frames=args.time_between_frames,
        extensions=extensions,
        include_labeled=args.include_labeled,
    )
    napari.run()


if __name__ == "__main__":
    sys.exit(main())