# python/chuzzle_suite/tests/test_tas.py
"""The Mind Bender TAS: puzzle choice, step order, trophy pop-ups per option, straight drags,
and the runner's input sequence (with the Windows input calls replaced by a recorder).
Run: python -m unittest discover -s python/chuzzle_suite/tests"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chuzzle_suite import board_sim, game_window as gw, levels as lv  # noqa: E402
from chuzzle_suite import tas_plan as tp, tas_runner as tr  # noqa: E402

PUZZLER, PSYCHIC, MASTER, SUPREMO, COLLECTOR = 21, 22, 23, 24, 25


def kinds(plan):
    return [s.kind for s in plan.steps]


def popups(plan):
    return [s.trophy for s in plan.steps if s.kind == "trophy"]


class PuzzleChoice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = lv.load_levels()

    def test_fewest_moves_win(self):
        picks, missing = tp.choose_level(self.levels, 8, 3)
        self.assertEqual([p.puzzle for p in picks], ["8-2", "8-4", "8-5"])     # 8 moves each
        self.assertEqual(missing, [])
        picks, _ = tp.choose_level(self.levels, 9, 3)
        self.assertEqual([p.puzzle for p in picks], ["9-1", "9-4", "9-5"])     # 8 + 7 + 9

    def test_puzzle_1_wins_a_tie_because_it_needs_no_click(self):
        # 5-1 + 5-2 + 5-3 and 5-2 + 5-3 + 5-4 are both 23 moves; 5-1 loads with the level
        picks, _ = tp.choose_level(self.levels, 5, 3)
        self.assertEqual([p.puzzle for p in picks], ["5-1", "5-2", "5-3"])

    def test_a_shorter_2x2_placement_is_used(self):
        pick = tp.best_pick(self.levels, "4-4")
        self.assertEqual((pick.goal, pick.move_count), ("4-4@r3c4", 6))     # the official goal is 7
        self.assertEqual(tp.best_pick(self.levels, "9-1").goal, "9-1")         # a tie keeps the official one

    def test_all_five_in_order(self):
        picks, _ = tp.choose_level(self.levels, 6, 5)
        self.assertEqual([p.number for p in picks], [1, 2, 3, 4, 5])

    def test_every_pick_solves_its_goal(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(target=SUPREMO))
        self.assertEqual(plan.puzzles, 100)
        for picks in plan.picks.values():
            for pick in picks:
                goal = lv.find_level(self.levels, pick.goal)
                end = board_sim.play(board_sim.BoardState(list(goal.start), goal.start_fat), pick.moves)[-1]
                self.assertEqual(end.cells, goal.goal, pick.goal)
                self.assertEqual(end.fat, goal.goal_fat, pick.goal)


class StepOrder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = lv.load_levels()

    def test_puzzler_run(self):
        plan = tp.build_plan(self.levels, tp.TasOptions())
        self.assertEqual(plan.options.per_level, 3)
        self.assertEqual(plan.options.last_level, 9)
        self.assertEqual(plan.puzzles, 27)
        self.assertEqual(plan.moves, 178)
        steps = plan.steps
        self.assertEqual(steps[0].kind, "split")
        self.assertEqual((steps[1].kind, steps[1].how, steps[1].puzzle), ("open", "auto", 1))
        arrows = [s.level for s in steps if s.kind == "arrow"]
        self.assertEqual(arrows, list(range(2, 11)))                  # ends with the slide into level 10
        final = [i for i, s in enumerate(steps) if s.final]
        self.assertEqual(len(final), 1)
        self.assertEqual((steps[final[0]].kind, steps[final[0]].level), ("arrow", 10))
        self.assertEqual(steps[final[0] + 1].kind, "split")           # the timer stops at the trophy
        self.assertEqual(kinds(plan)[-1], "trophy")                   # then its pop-up, nothing else

    def test_how_each_puzzle_opens(self):
        plan = tp.build_plan(self.levels, tp.TasOptions())
        opens = {(s.level, s.puzzle): s.how for s in plan.steps if s.kind == "open"}
        self.assertEqual(opens[(2, 1)], "auto")       # the level loads puzzle 1
        self.assertEqual(opens[(2, 4)], "star")       # 2-1 -> 2-4 skips puzzles: its star
        self.assertEqual(opens[(2, 5)], "next")       # 2-4 -> 2-5: the band of the solved puzzle
        self.assertEqual(opens[(8, 2)], "star")       # the first puzzle of a level, not puzzle 1
        # each level: open, solve, open, solve, open, solve, then its arrow
        level3 = [s.kind for s in plan.steps if s.level == 3 and s.kind in ("open", "solve")]
        self.assertEqual(level3, ["open", "solve"] * 3)

    def test_solve_steps_carry_their_drags(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(target=PSYCHIC))
        for s in plan.steps:
            if s.kind == "solve":
                self.assertEqual(len(tp.drags_for(self.levels, s.pick)), s.pick.move_count)

    def test_22_and_24_play_every_puzzle(self):
        for target, last in ((PSYCHIC, 10), (SUPREMO, 20)):
            plan = tp.build_plan(self.levels, tp.TasOptions(target=target, per_level=3))
            self.assertEqual((plan.options.per_level, plan.options.last_level), (5, last))
            self.assertEqual(plan.puzzles, 5 * last)
            final = [s for s in plan.steps if s.final]
            self.assertEqual([(s.kind, s.level, s.puzzle) for s in final], [("solve", last, 5)])

    def test_mentalist_master_ends_entering_level_20(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(target=MASTER))
        self.assertEqual(plan.puzzles, 57)
        final = [s for s in plan.steps if s.final]
        self.assertEqual([(s.kind, s.level) for s in final], [("arrow", 20)])

    def test_shorter_practice_run(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(last_level=3))
        self.assertEqual(plan.puzzles, 9)
        self.assertFalse(any(s.final for s in plan.steps))
        self.assertEqual([s.level for s in plan.steps if s.kind == "arrow"], [2, 3])   # no arrow after 3
        self.assertEqual(kinds(plan)[-1], "split")
        self.assertEqual(popups(plan), [])

    def test_without_livesplit(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(), livesplit=False)
        self.assertNotIn("split", kinds(plan))
        self.assertEqual(kinds(plan)[-1], "trophy")


class TrophyPopups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = lv.load_levels()

    def plan(self, **kw):
        return tp.build_plan(self.levels, tp.TasOptions(**kw))

    def test_new_profile_gets_the_pop_up(self):
        plan = self.plan()
        self.assertEqual(popups(plan), [PUZZLER])
        self.assertEqual(plan.earned, [PUZZLER])

    def test_owned_trophy_has_no_pop_up(self):
        plan = self.plan(owned=frozenset({PUZZLER}))
        self.assertEqual(popups(plan), [])
        self.assertEqual(kinds(plan)[-2:], ["arrow", "split"])
        self.assertTrue(plan.steps[-2].final)                         # still the run's end

    def test_mentalist_master_passes_puzzler(self):
        plan = self.plan(target=MASTER)
        self.assertEqual(popups(plan), [PUZZLER, MASTER])
        i = kinds(plan).index("trophy")
        self.assertEqual((plan.steps[i - 1].kind, plan.steps[i - 1].level), ("arrow", 10))
        self.assertEqual(plan.steps[i + 1].kind, "open")              # level 10 goes on after it

    def test_all_five_through_level_10_earns_psychic_on_the_way(self):
        plan = self.plan(target=MASTER, per_level=5)
        self.assertEqual(popups(plan), [PUZZLER, PSYCHIC, MASTER])
        i = [s.trophy for s in plan.steps if s.kind == "trophy"].index(PSYCHIC)
        step = [s for s in plan.steps if s.kind == "trophy"][i]
        before = plan.steps[plan.steps.index(step) - 1]
        self.assertEqual((before.kind, before.level, before.puzzle), ("solve", 10, 5))
        self.assertEqual(before.trophy, PSYCHIC)

    def test_psychic_run(self):
        self.assertEqual(popups(self.plan(target=PSYCHIC)), [PUZZLER, PSYCHIC])
        self.assertEqual(popups(self.plan(target=PSYCHIC, owned=frozenset({PUZZLER}))), [PSYCHIC])
        self.assertEqual(popups(self.plan(target=PSYCHIC, owned=frozenset({PUZZLER, PSYCHIC}))), [])

    def test_supremo_with_and_without_pop_ups(self):
        self.assertEqual(popups(self.plan(target=SUPREMO)), [PUZZLER, PSYCHIC, MASTER, SUPREMO])
        none = self.plan(target=SUPREMO, owned=frozenset({PUZZLER, PSYCHIC, MASTER, SUPREMO}))
        self.assertEqual(popups(none), [])
        self.assertEqual(none.earned, [])

    def test_trophy_collector_follows_the_last_mind_bender_trophy(self):
        plan = self.plan(target=SUPREMO, others_owned=True)
        self.assertEqual(popups(plan), [PUZZLER, PSYCHIC, MASTER, SUPREMO, COLLECTOR])
        # it comes with whichever Mind Bender trophy is last, here PUZZLER
        plan = self.plan(owned=frozenset({PSYCHIC, MASTER, SUPREMO}), others_owned=True)
        self.assertEqual(popups(plan), [PUZZLER, COLLECTOR])
        plan = self.plan(owned=frozenset({PSYCHIC, MASTER, SUPREMO, COLLECTOR}), others_owned=True)
        self.assertEqual(popups(plan), [PUZZLER])

    def test_estimate_is_the_timer_time(self):
        cost = tp.CostModel(move_s=0.1, click_s=0.05, puzzle_s=1.0, slide_s=0.3, trophy_s=2.0)
        plan = tp.build_plan(self.levels, tp.TasOptions(), cost=cost)
        clicks = plan.clicks - 1                              # the final pop-up comes after the timer
        self.assertAlmostEqual(plan.estimate_s(), 178 * 0.1 + 27 * 1.0 + clicks * 0.05 + 0.3)
        # a pop-up mid-run counts; the target's own pop-up (after the timer stops) does not
        with_pop = tp.build_plan(self.levels, tp.TasOptions(target=MASTER), cost=cost)
        without = tp.build_plan(self.levels, tp.TasOptions(target=MASTER, owned=frozenset({PUZZLER})), cost=cost)
        self.assertAlmostEqual(with_pop.estimate_s() - without.estimate_s(), 2.0)


class StraightDrags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.levels = lv.load_levels()

    def check(self, goal_name, moves):
        goal = lv.find_level(self.levels, goal_name)
        state = board_sim.BoardState(list(goal.start), goal.start_fat)
        for d in tp.straight_drags(state, moves):
            axis, lines, amount = board_sim.parse_token(d.token)
            self.assertEqual(d.cells % 6, amount, d)
            for x, y in (d.press, d.release):
                self.assertTrue(0 <= x <= 5 and 0 <= y <= 5, d)
            lane = d.press[1] if d.axis == "R" else d.press[0]
            self.assertIn(lane, lines, d)
            if state.fat is not None and len(lines) == 2:
                self.assertEqual(d.press, state.fat)                  # pressed on the 2x2 piece
            else:
                self.assertLessEqual(abs(d.cells), 3)
                self.assertNotEqual(d.cells, 0)
            state = board_sim.apply_move(state, d.token)
        self.assertEqual(state.cells, goal.goal)

    def test_every_planned_drag(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(target=SUPREMO))
        for picks in plan.picks.values():
            for pick in picks:
                self.check(pick.goal, pick.moves)

    def test_2x2_moves_keep_the_piece_on_the_board(self):
        level = self.levels["4-4"]
        for sol in level.solutions:
            self.check("4-4", sol)


class FakeWindow:
    hwnd = 1
    title = "Chuzzle Christmas Edition 1.0"
    exe = None

    def alive(self):
        return True

    def is_foreground(self):
        return True

    def client_rect(self):
        return FakeGame.LEFT, FakeGame.TOP, 800, 600

    def bring_to_front(self):
        return True


class FakeGame:
    """Mind Bender as the decomp describes it, fed the runner's SendInput stream: a drag starts at
    the first move more than 2 px from the press (row if |dx| > |dy|), its offset counts from
    there, the release snaps it; buttons fire on release when hovered; the banner of a just-solved
    puzzle goes to the next unsolved star; the arrow opens the next level once 3 are solved; a
    trophy pop-up blocks input until a click, then the trophy room until Back. Windows merges
    cursor moves the game has not read: the game reads its queue at most every `read_period`
    seconds, and a move queued right after another unread move replaces it (unless another input
    came between them and `keys_separate`)."""

    LEFT, TOP = 100, 50

    def __init__(self, levels, read_period=0.0, keys_separate=True, owned=(), level=1):
        self.levels = levels
        self.read_period = read_period
        self.keys_separate = keys_separate
        self.owned = set(owned)
        self.queue = []
        self.last_read = 0.0
        self.level, self.unlocked = level, level
        self.solved = set()
        self.splits = 0
        self.popups = []          # trophies shown
        self.modal = None         # "popup" | "room"
        self.x = self.y = 0
        self.press = None
        self.drag = None          # [axis, lane, last_x, last_y, offset]
        self.pressed_button = None
        self.ignored_while_modal = 0
        self.load(1)

    # --- state ---

    def name(self):
        return f"{self.level}-{self.slot + 1}"

    def load(self, number):
        self.slot = number - 1
        level = self.levels[self.name()]
        if self.name() in self.solved:
            self.board = board_sim.BoardState(list(level.goal), level.goal_fat)
            self.is_solved, self.banner, self.just_solved = True, True, False
        else:
            self.board = board_sim.BoardState(list(level.start), level.start_fat)
            self.is_solved, self.banner, self.just_solved = False, False, False

    def award(self, trophy):
        if trophy not in self.owned:
            self.owned.add(trophy)
            self.popups.append(trophy)
            self.modal = "popup"

    def solve_check(self):
        if self.board.cells != self.levels[self.name()].goal:
            return
        self.solved.add(self.name())
        self.is_solved, self.banner, self.just_solved = True, True, True
        if sum(f"{self.level}-{n}" in self.solved for n in range(1, 6)) >= 3:
            self.unlocked = max(self.unlocked, self.level + 1)
        if all(f"{w}-{n}" in self.solved for w in range(1, 11) for n in range(1, 6)):
            self.award(PSYCHIC)
        if len(self.solved) == 100:
            self.award(SUPREMO)

    def next_level(self):
        if self.level < self.unlocked and self.level < 20:
            self.level += 1
            self.load(1)
            if self.level >= 10:
                self.award(PUZZLER)
            if self.level >= 20:
                self.award(MASTER)

    # --- widgets (logical px) ---

    def widget_at(self, x, y):
        if self.modal == "room":
            return "back" if 280 <= x < 547 and 533 <= y < 593 else None
        if self.banner and 315 <= x < 678 and 317 <= y < 348:
            return "banner"
        if 169 <= x < 227 and 452 <= y < 513:
            return "arrow"
        for i, (cx, cy) in enumerate(tr.STARS):
            if (x - cx) ** 2 + (y - cy) ** 2 <= 20 ** 2:
                return f"star{i + 1}"
        if 255 <= x < 735 and 57 <= y < 537:
            return "board"
        return None

    def fire(self, widget):
        if widget == "back":
            self.modal = None
        elif widget == "arrow":
            self.next_level()
        elif widget and widget.startswith("star"):
            self.load(int(widget[4:]))
        elif widget == "banner":
            if self.just_solved:
                rest = [n for n in range(self.slot + 2, 6) if f"{self.level}-{n}" not in self.solved]
                rest = rest or [n for n in range(1, 6) if f"{self.level}-{n}" not in self.solved]
                if rest:
                    self.load(rest[0])
                else:
                    self.next_level()
            else:
                level = self.levels[self.name()]
                self.board = board_sim.BoardState(list(level.start), level.start_fat)
                self.is_solved = self.banner = False

    # --- input ---

    def send(self, events):
        now = time.perf_counter()
        if now - self.last_read >= self.read_period:
            self.read()
            self.last_read = now
        for e in events:
            last = self.queue[-1] if self.queue else None
            if e[0] == "key" and not self.keys_separate and e[1] == tr.VK_F24:
                continue
            if e[0] == "move" and last is not None and last[0] == "move":
                self.queue[-1] = e
            else:
                self.queue.append(e)

    def read(self):
        queue, self.queue = self.queue, []
        for e in queue:
            self.handle(e)

    def handle(self, e):
        kind = e[0]
        if kind == "key":
            if e[1] == tr.VK_NUMPAD1:
                self.splits += 1
            return
        if kind == "move":
            self.x, self.y = e[1] - self.LEFT, e[2] - self.TOP
            if self.press is not None:
                self.mouse_drag()
            return
        if kind == "down":
            if self.modal == "popup":
                self.modal = "room"
                return
            widget = self.widget_at(self.x, self.y)
            if self.modal == "room" and widget != "back":
                self.ignored_while_modal += 1
                return
            if widget == "board":
                if not self.is_solved:
                    self.press, self.drag = (self.x, self.y), None
            else:
                self.pressed_button = widget
            return
        # up
        if self.press is not None:
            if self.drag is not None:
                self.end_drag()
            self.press = self.drag = None
        elif self.pressed_button is not None:
            if self.widget_at(self.x, self.y) == self.pressed_button:     # hovered at release
                self.fire(self.pressed_button)
            self.pressed_button = None

    def mouse_drag(self):
        if self.drag is None:
            dx, dy = self.x - self.press[0], self.y - self.press[1]
            if dx * dx + dy * dy <= 4:
                return
            axis = "R" if abs(dx) > abs(dy) else "C"
            lane = (self.y - 57) // 80 if axis == "R" else (self.x - 255) // 80
            self.drag = [axis, lane, self.x, self.y, 0]
            return
        axis, lane, lx, ly, off = self.drag
        off += (self.x - lx) if axis == "R" else (self.y - ly)
        fat = self.board.fat
        if fat is not None and lane in ((fat[1], fat[1] + 1) if axis == "R" else (fat[0], fat[0] + 1)):
            left = fat[0] if axis == "R" else fat[1]
            off = max(-left * 80, min((4 - left) * 80, off))
        self.drag = [axis, lane, self.x, self.y, off]

    def end_drag(self):
        axis, lane, _, _, off = self.drag
        n = (int(off) + 40) // 80 if off >= 0 else -((-int(off) + 40) // 80)
        if n % 6 == 0:
            return
        fat = self.board.fat
        lines = [lane]
        if fat is not None and lane in ((fat[1], fat[1] + 1) if axis == "R" else (fat[0], fat[0] + 1)):
            lines = [fat[1], fat[1] + 1] if axis == "R" else [fat[0], fat[0] + 1]
        token = axis + "".join(map(str, lines)) + str(n % 6)
        self.board = board_sim.apply_move(self.board, token)
        self.solve_check()

    # --- screen ---

    def capture(self, window=None):
        self.read()
        buf = bytearray(800 * 600 * 4)
        for i, colour in enumerate(self.board.cells):
            r, c = divmod(i, 6)
            cx, cy = 295 + 80 * c, 97 + 80 * r
            red, green, blue = tr.PALETTE[colour]
            for px, py in tr.FUR_PATCHES:
                for dx in (-2, 0, 2):
                    for dy in (-2, 0, 2):
                        k = ((cy + py + dy) * 800 + cx + px + dx) * 4
                        buf[k:k + 3] = bytes((blue, green, red))
        for n, (cx, cy) in enumerate(tr.STARS):
            colour = (61, 173, 36) if f"{self.level}-{n + 1}" in self.solved else (74, 63, 54)   # BGR
            for dx, dy in tr.STAR_SAMPLES:
                k = ((cy + dy) * 800 + cx + dx) * 4
                buf[k:k + 3] = bytes(colour)
        return bytes(buf), 800, 600


class FakeGameRuns(unittest.TestCase):
    """The runner against the fake game: whole runs, trophy pop-ups, checks, Esc, measuring."""

    @classmethod
    def setUpClass(cls):
        cls.levels = lv.load_levels()

    def run_plan(self, plan, game, timing=None, **kw):
        timing = timing or tr.TasTiming(key_break=True, gap=0.0, pace=0.0, margin=0.0)
        done = []
        with self.patched(game):
            runner = tr.TasRunner(FakeWindow(), gw.Geometry(), self.levels, plan, timing,
                                  on_done=lambda msg, secs: done.append(msg), **kw)
            runner._run()
        game.read()
        return done[0]

    def patched(self, game, esc=lambda vk: False):
        from contextlib import ExitStack
        stack = ExitStack()
        for target, attr, value in [
                (tr, "send", game.send), (gw, "capture_client", game.capture),
                (gw, "looks_paused", lambda w: False), (gw, "key_down", esc),
                (tr, "POPUP_AFTER_ARROW_S", 0.0), (tr, "POPUP_AFTER_SOLVE_S", 0.0), (tr, "ROOM_IN_S", 0.0), (tr, "ROOM_OUT_S", 0.0), (tr, "HOVER_S", 0.0), (tr, "SLIDE_S", 0.0)]:
            stack.enter_context(mock.patch.object(target, attr, value))
        return stack

    def test_puzzler_run(self):
        plan = tp.build_plan(self.levels, tp.TasOptions())
        game = FakeGame(self.levels)
        self.assertEqual(self.run_plan(plan, game), "done")
        self.assertEqual(game.level, 10)                       # reached level 10
        picked = {p.puzzle for ps in plan.picks.values() for p in ps}
        self.assertEqual(game.solved, picked)                  # 4-4 with the 2x2 piece elsewhere too
        self.assertEqual(game.splits, 2)
        self.assertEqual(game.popups, [PUZZLER])
        self.assertIsNone(game.modal)                          # its pop-up clicked away
        self.assertEqual(game.ignored_while_modal, 0)

    def test_psychic_run_clicks_the_puzzler_pop_up_away_mid_run(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(target=PSYCHIC))
        game = FakeGame(self.levels)
        self.assertEqual(self.run_plan(plan, game), "done")
        self.assertEqual(len(game.solved), 50)
        self.assertEqual(game.popups, [PUZZLER, PSYCHIC])

    def test_owned_trophies_no_pop_ups(self):
        owned = frozenset({PUZZLER, PSYCHIC})
        plan = tp.build_plan(self.levels, tp.TasOptions(target=PSYCHIC, owned=owned))
        game = FakeGame(self.levels, owned=owned)
        self.assertEqual(self.run_plan(plan, game), "done")
        self.assertEqual((len(game.solved), game.popups), (50, []))

    def test_gap_instead_of_the_key(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(last_level=2))
        game = FakeGame(self.levels, keys_separate=False)
        timing = tr.TasTiming(key_break=False, gap=0.0, pace=0.0)
        self.assertEqual(self.run_plan(plan, game, timing), "done")
        self.assertEqual(len(game.solved), 6)

    def test_merged_moves_are_caught(self):
        # the key does not keep the moves apart here: the nudge and the jump merge, nothing moves
        plan = tp.build_plan(self.levels, tp.TasOptions(last_level=2))
        game = FakeGame(self.levels, keys_separate=False)
        with mock.patch.object(tr, "SOLVE_CHECK_GRACE_S", 0.05):
            msg = self.run_plan(plan, game)
        self.assertEqual(msg, "stopped: 1-1, 1-2, 1-3 did not register as solved (inputs too fast? measure the speed)")
        self.assertEqual(game.solved, set())

    def test_the_run_must_start_on_1_1(self):
        plan = tp.build_plan(self.levels, tp.TasOptions())
        game = FakeGame(self.levels, level=3)
        msg = self.run_plan(plan, game)
        self.assertEqual(msg, "stopped: the game should show 1-1 scrambled, but shows 3-1 (scrambled)")
        self.assertEqual(game.splits, 0)                       # nothing was sent

    def test_without_checks_it_does_not_read_the_screen(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(last_level=1))
        game = FakeGame(self.levels)
        game.capture = mock.Mock(side_effect=AssertionError("read the screen"))
        self.assertEqual(self.run_plan(plan, game, check_solves=False), "done")
        self.assertEqual(len(game.solved), 3)

    def test_esc_stops_and_releases_the_button(self):
        plan = tp.build_plan(self.levels, tp.TasOptions(last_level=2), livesplit=False)
        game = FakeGame(self.levels, keys_separate=False)
        sent = []
        calls = {"n": 0}

        def esc(vk):
            calls["n"] += 1
            return calls["n"] > 12                             # Esc goes down mid-run

        def record(events):
            sent.extend(e[0] for e in events)
            game.send(events)

        timing = tr.TasTiming(key_break=False, gap=0.0, pace=0.0)
        done = []
        with self.patched(game, esc), mock.patch.object(tr, "send", record):
            runner = tr.TasRunner(FakeWindow(), gw.Geometry(), self.levels, plan, timing,
                                  check_solves=False, on_done=lambda msg, secs: done.append(msg))
            runner._run()
        self.assertEqual(done, ["stopped: Esc pressed"])
        self.assertEqual(sent.count("down"), sent.count("up"))
        self.assertEqual(sent[-1], "up")

    def test_measure_finds_the_key(self):
        game = FakeGame(self.levels, read_period=0.01, level=2)
        game.load(3)                                          # 2-3, 6 moves
        result = []
        with self.patched(game):
            m = tr.SpeedMeasure(FakeWindow(), gw.Geometry(), self.levels, tr.TasTiming(),
                                on_done=lambda msg, t: result.append((msg, t)))
            m._run()
        msg, timing = result[0]
        self.assertIsNotNone(timing, msg)
        self.assertTrue(timing.key_break and timing.measured, msg)
        self.assertEqual(timing.gap, 0.0)

    def test_measure_finds_the_shortest_gap(self):
        game = FakeGame(self.levels, read_period=0.01, keys_separate=False, level=2)
        game.load(3)
        result = []
        with self.patched(game):
            m = tr.SpeedMeasure(FakeWindow(), gw.Geometry(), self.levels, tr.TasTiming(),
                                on_done=lambda msg, t: result.append((msg, t)))
            m._run()
        msg, timing = result[0]
        self.assertIsNotNone(timing, msg)
        self.assertFalse(timing.key_break)
        self.assertTrue(0.008 <= timing.gap <= 0.04, msg)
        self.assertIn("2-3", msg)


class ScreenReading(unittest.TestCase):
    def test_fur_colours_by_hue(self):
        # the Christmas Edition's fur, Deluxe's more saturated fur, and what is no chuzzle
        self.assertEqual([tr.fur_colour(c) for c in tr.PALETTE], list(range(8)))
        self.assertEqual(tr.fur_colour((12, 178, 12)), 1)
        self.assertEqual(tr.fur_colour((218, 37, 37)), 0)
        self.assertIsNone(tr.fur_colour((20, 20, 30)))           # the dark board between chuzzles
        self.assertIsNone(tr.fur_colour((120, 140, 120)))        # greyish: no colour


class ProfileFiles(unittest.TestCase):
    """INFO.CFG trophies and the Mind Bender save, for the tab's "Read profile" and save line."""

    def test_owned_trophies(self):
        from chuzzle_suite import saves
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "INFO.CFG").write_text("\n".join(
                ["TotalPopped=123", "T21=Received 11/08/26 at 10:00 AM", "T3=Received x", ""]))
            self.assertEqual(saves.owned_trophies(folder), {3, 21})
            self.assertEqual(saves.owned_trophies(folder / "missing"), set())

    def test_mindbender_save(self):
        import struct
        from chuzzle_suite import saves
        data = bytearray(1985)
        struct.pack_into("<Ii5s", data, 0, 0xBEAD, 5, b"1.24\x00")
        struct.pack_into("<i", data, 46, 4)                       # level shown
        for lvl, slot in ((0, 0), (0, 1), (2, 4), (19, 4)):     # 1-1, 1-2, 3-5, 20-5
            data[132 + slot * 20 + lvl] = 1
        struct.pack_into("<i", data, 236, 5)                      # levels open
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / saves.MINDBENDER_SAVE
            path.write_bytes(bytes(data))
            s = saves.read_mindbender_save(path)
            self.assertEqual((s.level, s.solved, s.unlocked), (4, {"1-1", "1-2", "3-5", "20-5"}, 5))
            path.write_bytes(b"something else" * 40)
            self.assertIsNone(saves.read_mindbender_save(path))
            self.assertIsNone(saves.read_mindbender_save(Path(tmp) / "none.dat"))


class TimingFile(unittest.TestCase):
    def test_round_trip_per_exe(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(tr, "TIMING_FILE", Path(tmp) / "tas_timing.json"):
            w = FakeWindow()
            self.assertFalse(tr.load_timing(w).measured)
            tr.save_timing(w, tr.TasTiming(gap=0.02, measured=True))
            self.assertEqual(tr.load_timing(w).gap, 0.02)
            self.assertFalse(tr.load_timing(None).measured)          # another exe keeps the default
            tr.clear_timing(w)
            self.assertFalse(tr.load_timing(w).measured)


if __name__ == "__main__":
    unittest.main()
