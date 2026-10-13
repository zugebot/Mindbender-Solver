# python/demo_app/tests/test_core.py
"""Run: python -m unittest discover -s python/demo_app/tests"""

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chuzzle_demo import board_sim, game_window as gw, launcher, levels as lv, routes, saves  # noqa: E402


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

    def test_unsolved_bounds_come_from_build_readme(self):
        # the solver queue raises these over time, so compare with the file itself
        import re
        text = lv.paths.BUILD_README.read_text(errors="replace")
        bounds = {n: int(b) for n, b in re.findall(r"^\s*(\d+-\d+): at least (\d+) moves", text, re.M)}
        self.assertIn("6-4", bounds)
        self.assertGreaterEqual(bounds["6-4"], 16)                 # none up to 15
        for name, bound in bounds.items():
            self.assertEqual(self.levels[name].lower_bound, bound, name)
        self.assertIsNone(self.levels["10-2"].lower_bound)        # solved

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


class Launcher(unittest.TestCase):
    def test_labels_are_the_shortest_distinct_folder_tail(self):
        exes = [Path(r"C:\Program Files (x86)\PopCap Games\Chuzzle\Chuzzle.exe"),
                Path(r"E:\Games\Chuzzles\Chuzzle\Chuzzle.exe"),
                Path(r"E:\Games\Chuzzles\Chuzzle\Chuzzle-WT.exe"),
                Path(r"E:\Games\Chuzzle (Christmas Edition)\Chuzzle.exe")]
        labels = launcher.unique_labels(exes)
        self.assertEqual(labels[exes[0]], r"PopCap Games\Chuzzle")
        self.assertEqual(labels[exes[1]], r"Chuzzles\Chuzzle")
        self.assertEqual(labels[exes[2]], r"Chuzzles\Chuzzle  (Chuzzle-WT.exe)")
        self.assertEqual(labels[exes[3]], "Chuzzle (Christmas Edition)")
        self.assertEqual(len(set(labels.values())), len(exes))

    def test_full_scan_finds_game_exes_and_skips_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ["Games/Chuzzle Deluxe/Chuzzle.exe", "Games/Chuzzle Deluxe/PopUninstall.exe",
                        "Games/Other/chuzzle_xmas.exe", "Temp/copy/Chuzzle.exe", "node_modules/x/chuzzle.exe",
                        "Games/Chuzzle Deluxe/readme.txt"]:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_bytes(b"MZ")
            found = {p.relative_to(root).as_posix() for p in launcher.full_scan(roots=[root])}
            self.assertEqual(found, {"Games/Chuzzle Deluxe/Chuzzle.exe", "Games/Other/chuzzle_xmas.exe"})

    def test_describe_reads_build_date_and_drm_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "Chuzzle.exe"
            header = bytearray(0x100)
            header[0:2] = b"MZ"
            header[0x3C:0x40] = (0x80).to_bytes(4, "little")
            header[0x80:0x84] = b"PE\0\0"
            header[0x88:0x8C] = (1156198680).to_bytes(4, "little")   # 2006-08-21 22:18 UTC
            exe.write_bytes(bytes(header) + launcher.DRM_MARKER + b"rest")
            g = launcher.describe_exe(exe)
            self.assertEqual(g.build_date, "2006-08-21")
            self.assertTrue(g.drm_wrapped)
            self.assertEqual(g.product_version, "")

    def test_hollow_shell_is_flagged(self):
        def pe_with_code(code: bytes) -> bytes:
            data = bytearray(0x200)
            data[0:2] = b"MZ"
            data[0x3C:0x40] = (0x80).to_bytes(4, "little")
            data[0x80:0x84] = b"PE\0\0"
            data[0x86:0x88] = (1).to_bytes(2, "little")         # one section
            data[0x94:0x96] = (0).to_bytes(2, "little")         # no optional header
            sec = 0x98                                          # section table
            data[sec:sec + 5] = b".text"
            data[sec + 16:sec + 20] = len(code).to_bytes(4, "little")      # SizeOfRawData
            data[sec + 20:sec + 24] = (0x100).to_bytes(4, "little")        # PointerToRawData
            data[sec + 36:sec + 40] = (0x60000020).to_bytes(4, "little")   # code, execute, read
            data[0x100:0x100 + len(code)] = code
            return bytes(data)

        self.assertTrue(launcher._is_hollow(pe_with_code(bytes(64))))
        self.assertFalse(launcher._is_hollow(pe_with_code(b"\x55\x8b\xec" + bytes(61))))



class PlayerMotion(unittest.TestCase):
    """The glide between drags: a gentle arc to the exact press point, a hand-like speed profile."""

    def _player(self):
        from chuzzle_demo.player import Player

        class Win:
            def client_rect(self):
                return 0, 0, 800, 600

        p = Player.__new__(Player)
        p.window = Win()
        p.background = True
        p._pos = (100.0, 100.0)
        p._arc_side = 1.0
        return p

    def test_arc_ends_on_target_bows_sideways_and_alternates(self):
        import math
        p = self._player()
        a = p._approach((500.0, 100.0))
        self.assertEqual(a[-1], (500.0, 100.0))
        bow1 = max(abs(y - 100.0) for _, y in a)
        self.assertTrue(20 < bow1 <= 60 + 1e-9, bow1)          # ~12% of 400 px, capped at 60
        side1 = math.copysign(1, a[len(a) // 2][1] - 100.0)
        b = p._approach((500.0, 100.0))
        side2 = math.copysign(1, b[len(b) // 2][1] - 100.0)
        self.assertEqual(side1, -side2)                            # alternating sides
        self.assertEqual(p._approach((110.0, 105.0)), [(110.0, 105.0)])   # short hops stay straight

    def test_reach_easing_is_monotonic_and_front_loaded(self):
        from chuzzle_demo.player import Player
        vals = [Player._ease_reach(i / 100) for i in range(101)]
        self.assertEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[-1], 1.0)
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))
        self.assertGreater(Player._ease_reach(0.5), Player._ease(0.5))   # ahead of the symmetric curve

    def test_sprite_has_alpha_and_a_pressed_variant(self):
        import os
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from chuzzle_demo import cursor_overlay as co
        up, down = co.render_sprite(False), co.render_sprite(True)
        self.assertEqual(len(up), co.SIZE * co.SIZE * 4)
        self.assertNotEqual(up, down)
        alphas = up[3::4]
        self.assertIn(0, alphas)                                   # transparent around the arrow
        self.assertIn(255, alphas)                                 # opaque arrow
        hx, hy = co.HOTSPOT
        self.assertGreater(up[(hy * co.SIZE + hx) * 4 + 3], 0)      # the tip is drawn at the hotspot

if __name__ == "__main__":
    unittest.main()
