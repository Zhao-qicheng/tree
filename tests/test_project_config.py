from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_config import PROJECT_ROOT, get_path, get_setting, load_dotenv


class TestProjectConfig(unittest.TestCase):
    def test_load_dotenv_preserves_process_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "# comment\nTEST_PROJECT_VALUE=from-file\n"
                "TEST_PROJECT_QUOTED='quoted value'\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"TEST_PROJECT_VALUE": "from-process"}, clear=False
            ):
                os.environ.pop("TEST_PROJECT_QUOTED", None)
                load_dotenv(env_file)
                self.assertEqual(get_setting("TEST_PROJECT_VALUE", "missing"), "from-process")
                self.assertEqual(get_setting("TEST_PROJECT_QUOTED", "missing"), "quoted value")

    def test_get_path_resolves_relative_values_from_project_root(self):
        with patch.dict(
            os.environ, {"TEST_PROJECT_PATH": "relative/data"}, clear=False
        ):
            self.assertEqual(
                get_path("TEST_PROJECT_PATH", "unused"),
                (PROJECT_ROOT / "relative" / "data").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
