# python/demo_app/tests/test_core.py
"""Run: python -m unittest discover -s python/demo_app/tests"""

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chuzzle_demo import board_sim, game_window as gw, levels as lv, routes, saves  # noqa: E402


class LevelsAndSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = lv.load_levels()

    def test_all_100_levels_load(self):
        names = {f"{w}-{p}" for w in range(1, 21) for p in range(1, 6)}
        self.assertTrue(names <= set(self.levels))

    def test_every_stored_solution_reaches_the_goal(self):
        checked = 0
        for level in self.levels.values():
            start = board_sim.BoardState(list(level.start), level.start_fat)
            for sol in level.solutions:
                end = board_sim.play(start, sol)[-1]
                self.assertEqual(end.cells, level.goal, f"{level.name}: {sol}")
                self.assertEqual(end.fat, level.goal_fat, f"{level.name}: {sol}")
                self.assertEqual(len(sol.split()), level.optimum, f"{level.name}: {sol}")
                checked += 1
        self.assertGreater(checked, 90000)

    def test_display_is_one_indexed(self):
        self.assertEqual(lv.to_display("R04 C342 R50"), "R14 C452 R60")

    def test_fat_piece_cannot_cross_edge_or_split(self):
        state = board_sim.BoardState([0] * 36, (3, 2))
        with self.assertRaises(board_sim.IllegalMove):
            board_sim.apply_move(state, "R232")          # x 3 -> 5: straddles the edge
        with self.assertRaises(board_sim.IllegalMove):
            board_sim.apply_move(state, "R21")           # only one of its rows
        self.assertEqual(board_sim.apply_move(state, "R234").fat, (1, 2))   # wraps to x 1

    def test_simple_route_drags_whole_lane_distance(self):
        level = self.levels["6-2"]
        route = routes._simple_route(level, level.solutions[0])
        self.assertEqual(len(route.drags), level.optimum)
        for drag in route.drags:
            (x0, y0), (x1, y1) = drag.points[0], drag.points[-1]
            moved = (x1 - x0) if drag.move[0] == "R" else (y1 - y0)
            self.assertEqual(int(moved) % 6, int(drag.move[-1]) % 6, drag)

    @unittest.skipUnless(routes.backend_available(), "python/viewer mouse DLL not built")
    def test_library_route_matches_move_count(self):
        level = self.levels["12-3"]
        route = routes.route_for(level, level.solutions[0])
        self.assertEqual([d.move for d in route.drags], level.solutions[0].split())
        self.assertGreater(route.cost, 0)


class FakeWindow:
    def client_rect(self):
        return 100, 50, 1024, 768


class GeometryMapping(unittest.TestCase):
    def test_default_layout_scales_with_client_size(self):
        geo = gw.Geometry()
        # top-left chuzzle centre at logical (295, 100) of 800x600
        self.assertEqual(geo.to_screen(FakeWindow(), (0, 0)), (100 + round(295 * 1024 / 800), 50 + 128))
        x5, _ = geo.to_screen(FakeWindow(), (5, 0))
        self.assertEqual(x5, 100 + round((295 + 400) * 1024 / 800))

    def test_calibration_from_corners(self):
        geo = gw.Geometry.from_corners((0.3, 0.2), (0.8, 0.7))
        self.assertAlmostEqual(geo.cell[0], 0.1)
        self.assertEqual(geo.to_screen(FakeWindow(), (5, 5)), (100 + round(0.8 * 1024), 50 + round(0.7 * 768)))


class SaveLocator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.game = root / "Games" / "Chuzzle.Deluxe"
        profiles = self.game / "Profiles"
        (profiles / "Jerrin").mkdir(parents=True)
        (profiles / "Other").mkdir()
        (self.game / "Chuzzle.exe").write_bytes(b"MZ")
        (self.game / "PopUninstall.exe").write_bytes(b"MZ")
        (profiles / "config.cfg").write_text("Profiles=Jerrin,Other\r\nCurrentProfile=Jerrin\r\nDX::ScreenWidth=1024\r\n")
        (profiles / "HSSpeedChuzzle.cfg").write_text("NAME0=Jerrin\r\n")
        (profiles / "Jerrin" / "SAVEGAME-SPEEDCHUZZLE.DAT").write_bytes(b"\x01" * 16)
        (profiles / "Jerrin" / "SAVEGAME-MINDBENDER.DAT").write_bytes(b"\x02" * 16)
        (profiles / "Other" / "INFO.CFG").write_text("TotalPopped=1\r\n")
        (profiles / "notes").mkdir()            # no save files: not a profile
        self.backups = root / "backups"

    def tearDown(self):
        self.tmp.cleanup()

    def test_finds_exe_and_profiles_next_to_it(self):
        exes = saves.find_game_exes([self.game.parent])
        self.assertEqual(exes, [self.game / "Chuzzle.exe"])
        loc = saves.scan_location(saves.candidate_locations(exes)[0])
        self.assertEqual(loc.kind, "install")
        self.assertEqual(loc.current_profile, "Jerrin")
        self.assertEqual([p.name for p in loc.profiles], ["Jerrin", "Other"])
        self.assertEqual({f.path.name for f in loc.shared_files}, {"config.cfg", "HSSpeedChuzzle.cfg"})
        self.assertIs(saves.active_location([loc], self.game / "Chuzzle.exe"), loc)

    def test_virtualstore_only_for_protected_folders(self):
        self.assertIsNone(saves.virtualstore_mirror(Path(r"E:\Games\Chuzzle")))
        mirror = saves.virtualstore_mirror(Path(r"C:\Program Files (x86)\PopCap Games\Chuzzle"))
        self.assertTrue(str(mirror).endswith(r"VirtualStore\Program Files (x86)\PopCap Games\Chuzzle"))

    def test_watcher_reports_changes(self):
        loc = saves.scan_location(saves.candidate_locations([self.game / "Chuzzle.exe"])[0])
        watcher = saves.SaveWatcher([loc])
        target = self.game / "Profiles" / "Jerrin" / "SAVEGAME-MINDBENDER.DAT"
        target.write_bytes(b"\x03" * 20)
        (self.game / "Profiles" / "Other" / "SAVEGAME-ZENCHUZZLE.DAT").write_bytes(b"z")
        changes = {(e.change, e.path.name) for e in watcher.poll()}
        self.assertEqual(changes, {("changed", "SAVEGAME-MINDBENDER.DAT"), ("added", "SAVEGAME-ZENCHUZZLE.DAT")})

    def test_backup_restore_and_stash(self):
        folder = self.game / "Profiles" / "Jerrin"
        backup = saves.backup_folder(folder, "deluxe", self.backups)
        self.assertEqual(sorted(p.name for p in backup.iterdir()),
                         ["SAVEGAME-MINDBENDER.DAT", "SAVEGAME-SPEEDCHUZZLE.DAT"])
        (folder / "SAVEGAME-MINDBENDER.DAT").write_bytes(b"changed")
        time.sleep(1.1)                          # backups are named by the second
        safety = saves.restore_backup(backup, folder, "deluxe", self.backups)
        self.assertEqual((folder / "SAVEGAME-MINDBENDER.DAT").read_bytes(), b"\x02" * 16)
        self.assertEqual((safety / "SAVEGAME-MINDBENDER.DAT").read_bytes(), b"changed")
        self.assertEqual(saves.list_backups(folder, "deluxe", self.backups)[0], backup)

        stashed = saves.stash_file(folder / "SAVEGAME-SPEEDCHUZZLE.DAT", "speed", backup_root=self.backups)
        self.assertFalse((folder / "SAVEGAME-SPEEDCHUZZLE.DAT").exists())
        self.assertEqual(stashed.read_bytes(), b"\x01" * 16)
        self.assertIsNone(saves.stash_file(folder / "SAVEGAME-SPEEDCHUZZLE.DAT", "speed", backup_root=self.backups))


if __name__ == "__main__":
    unittest.main()
