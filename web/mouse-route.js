// web/mouse-route.js - the mouse route of a solution: for each move where to press, which way
// to drag and where to let go, with the least mouse travel over the whole solution.
//
// A port of mouse/src/chuzzle_mouse.c (chuzzle_solve_sequence_utf8, the library Chuzzle Suite
// plays and ranks game puzzles with; the C stays as it is). The same search:
//   - each move's candidates: every cell along its lane to press on (either lane of a 2x2
//     move), times both ways round (right n or left 6 - n); the 2x2 never crosses an edge, so
//     a 2x2 move keeps only the way that fits, tracked through the solution
//   - a drag of 2 or more cells is free once the lane locks (1 cell along it): only the motion
//     along the lane moves it, so the release may sit anywhere across the lane, and is put
//     where the way on to the next press is shortest
//   - dynamic programming over the moves (candidate x the 2x2's spot), least total travel
// Changes: any board size (the C's 6 is a constant; 5x5 boards are Chuzzle 2's); the release
// across the lane is found exactly (the mirror-image shortest path) instead of by a 70-step
// ternary search, the same point to float precision and far faster, so a puzzle's thousands
// of solutions can be ranked in the page; and `onBoard`: no press, lock or release past the
// board's edge (a phone cannot drag beyond the screen), which drops only the candidates whose
// drag would run off the board (every move keeps one that fits).
//
// Points are in cells, (column, row), integers at the cells' centres.
// plan(moves, { size, fat, onBoard }) -> { cost, steps: [{ d, lane, free, press, lock, release }] }
//   moves: [{ col, lanes, n }] (lanes 0-indexed; two lanes = the 2x2's), fat: the 2x2's
//   top-left at the start { x, y } or null; cost: the mouse travel in cells, the drags plus the
//   moves between them. null if no route exists (never, for a valid solution).
(function () {
  const LOCK = 1.0;          // ChuzzleSolveConfig.lock_threshold (the adapter's default)
  const FREE_MIN = 2;        // free_drag_min_displacement

  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

  function candidates(mv, N, onBoard) {
    const a = ((mv.n % N) + N) % N;
    const ds = a === 0 ? [0] : [a, -(N - a)];
    const out = [];
    for (const lane of mv.lanes) {
      for (let anchor = 0; anchor < N; anchor++) {
        for (const d of ds) {
          const free = Math.abs(d) >= FREE_MIN && Math.abs(d) > LOCK;
          const sign = Math.sign(d);
          const relAlong = anchor + d, lockAlong = free ? anchor + sign * LOCK : relAlong;
          if (onBoard && (relAlong < 0 || relAlong > N - 1 || lockAlong < 0 || lockAlong > N - 1)) continue;
          const pt = (along) => (mv.col ? { x: lane, y: along } : { x: along, y: lane });
          out.push({ lane, d, free, press: pt(anchor), lock: pt(lockAlong), rel: pt(relAlong), min: Math.abs(d) });
        }
      }
    }
    return out;
  }

  // The release of candidate c on the way to point t (chuzzle_optimize_release_to_point): a
  // free drag lets go on the line across the lane through its exact end, at the point of that
  // line nearest the path lock -> line -> t (the line is crossed where the straight path from
  // the lock to t's mirror image would cross it).
  function releaseTo(c, t, col) {
    if (!c.free) {
      const move = dist(c.rel, t);
      return { release: c.rel, drag: c.min, move, total: c.min + move };
    }
    let release;
    if (!col) {                                  // a row: the line x = rel.x
      const X = c.rel.x, p = Math.abs(c.lock.x - X), q = Math.abs(t.x - X);
      release = { x: X, y: p + q > 1e-12 ? c.lock.y + (t.y - c.lock.y) * p / (p + q) : Math.min(c.lock.y, t.y) };
    } else {                                     // a column: the line y = rel.y
      const Y = c.rel.y, p = Math.abs(c.lock.y - Y), q = Math.abs(t.y - Y);
      release = { x: p + q > 1e-12 ? c.lock.x + (t.x - c.lock.x) * p / (p + q) : Math.min(c.lock.x, t.x), y: Y };
    }
    const drag = dist(c.press, c.lock) + dist(c.lock, release), move = dist(release, t);
    return { release, drag, move, total: drag + move };
  }

  function plan(moves, { size = 6, fat = null, onBoard = true } = {}) {
    const N = size, M = moves.length;
    if (!M) return { cost: 0, steps: [] };
    const layers = moves.map((mv) => candidates(mv, N, onBoard));
    if (layers.some((l) => !l.length)) return null;
    const F = fat ? (N - 1) * (N - 1) : 1;
    // the 2x2's spot after candidate c of move i from spot f (-1: it would cross an edge)
    const fatAfter = (i, c, f) => {
      if (!fat) return 0;
      const mv = moves[i], x = f % (N - 1), y = Math.floor(f / (N - 1));
      if (mv.lanes.length !== 2) return f;
      const lo = Math.min(...mv.lanes), at = mv.col ? x : y;
      if (lo !== at) return f;                       // not this 2x2's lanes
      const next = (mv.col ? y : x) + c.d;
      if (next < 0 || next > N - 2) return -1;
      return mv.col ? next * (N - 1) + x : y * (N - 1) + next;
    };
    const f0 = fat ? fat.y * (N - 1) + fat.x : 0;
    // cost[c * F + f]: the least travel to the end of move i with candidate c, the 2x2 at f
    let cost = new Float64Array(layers[0].length * F).fill(Infinity);
    const parents = [null];
    layers[0].forEach((c, ci) => {
      const f = fatAfter(0, c, f0);
      if (f >= 0) cost[ci * F + f] = 0;
    });
    for (let i = 1; i < M; i++) {
      const prev = layers[i - 1], cur = layers[i], col = moves[i - 1].col;
      const edge = prev.map((p) => cur.map((c) => releaseTo(p, c.press, col).total));
      const next = new Float64Array(cur.length * F).fill(Infinity);
      const par = new Int32Array(cur.length * F).fill(-1);
      for (let pi = 0; pi < prev.length; pi++) {
        for (let pf = 0; pf < F; pf++) {
          const base = cost[pi * F + pf];
          if (base === Infinity) continue;
          for (let ci = 0; ci < cur.length; ci++) {
            const f = fatAfter(i, cur[ci], pf);
            if (f < 0) continue;
            const s = ci * F + f, v = base + edge[pi][ci];
            if (v < next[s]) {
              next[s] = v;
              par[s] = pi * F + pf;
            }
          }
        }
      }
      cost = next;
      parents.push(par);
    }
    // the last drag ends where it lands (no target after it)
    let best = Infinity, bestState = -1;
    const last = layers[M - 1];
    for (let ci = 0; ci < last.length; ci++) {
      for (let f = 0; f < F; f++) {
        const v = cost[ci * F + f] + last[ci].min;
        if (v < best) {
          best = v;
          bestState = ci * F + f;
        }
      }
    }
    if (bestState < 0) return null;
    const chosen = Array(M);
    for (let i = M - 1, s = bestState; i >= 0; i--) {
      chosen[i] = layers[i][Math.floor(s / F)];
      if (i > 0) s = parents[i][s];
    }
    const steps = chosen.map((c, i) => {
      const r = i + 1 < M ? releaseTo(c, chosen[i + 1].press, moves[i].col).release : c.rel;
      return { d: c.d, lane: c.lane, free: c.free, press: c.press, lock: c.lock, release: r };
    });
    return { cost: best, steps };
  }

  self.MB_ROUTE = { plan };
})();
