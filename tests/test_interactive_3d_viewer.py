import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_interactive_3d_viewer.py"


class _ScriptSourceCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "script":
            return
        attributes = dict(attrs)
        if "src" in attributes:
            self.sources.append(attributes["src"])


class TestInteractive3DViewerBuild(unittest.TestCase):
    def test_builds_self_contained_human17_region_viewer(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            frame_dir = temp_dir / "frames"
            frame_dir.mkdir()
            (frame_dir / "0000.jpg").write_bytes(b"test frame")
            (frame_dir / "0001.jpg").write_bytes(b"test frame")

            poses = np.zeros((2, 17, 3), dtype=np.float32)
            poses[:, :, 0] = np.linspace(-0.8, 0.8, 17)
            poses[:, :, 1] = np.linspace(0.0, 1.6, 17)
            poses[1, :, 2] = 0.1
            npz_path = temp_dir / "poses.npz"
            np.savez(npz_path, pred3d_root_relative_image_units=poses)

            html_path = temp_dir / "viewer.html"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BUILD_SCRIPT),
                    "--input-3d-npz",
                    str(npz_path),
                    "--left-frame-dir",
                    str(frame_dir),
                    "--out-html",
                    str(html_path),
                    "--fps",
                    "30",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("frames: 2", completed.stdout)
            generated = html_path.read_text(encoding="utf-8")
            self.assertIn("Human17RegionModel.createHuman17RegionViewer", generated)
            self.assertIn('id="regionToggle"', generated)
            self.assertIn('id="skeletonToggle"', generated)
            self.assertIn('id="regionThickness"', generated)

            placeholders = (
                "__TITLE__",
                "__REGION_MODEL_BUNDLE__",
                "__POSE_DATA__",
                "__PAIRS__",
                "__BONE_COLORS__",
                "__LEFT_DIR__",
                "__FPS__",
                "__RADIUS__",
                "__SKELETON_SCALE__",
                "__LAST_FRAME__",
            )
            for placeholder in placeholders:
                self.assertNotIn(placeholder, generated)

            parser = _ScriptSourceCollector()
            parser.feed(generated)
            self.assertEqual(parser.sources, [])
            self.assertNotIn("cdn.jsdelivr.net", generated)
            self.assertNotIn("unpkg.com", generated)


if __name__ == "__main__":
    unittest.main()
