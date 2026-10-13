import pathlib
import shutil
import subprocess
import unittest

GODOT = shutil.which('godot') or '/Applications/Godot.app/Contents/MacOS/Godot'
LANGO = pathlib.Path(__file__).resolve().parents[2] / 'examples' / 'Lango_GodotRPG'  # a sibling checkout of the Lango example


class OverlayLifetimeTests(unittest.TestCase):
    @unittest.skipUnless(pathlib.Path(GODOT).is_file() and LANGO.is_dir(), 'Godot or the Lango example checkout is missing')
    def test_deleted_nodes_do_not_raise_deferred_errors(self):
        result = subprocess.run([GODOT, '--headless', '--path', str(LANGO), '--script', 'res://Debug/tests/test_overlay_lifetime.gd'], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('ERROR:', result.stderr, result.stderr)
        self.assertIn('PASS:', result.stdout)
