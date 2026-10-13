// web/solver-worker.js - one solver (mb_solver.js, WebAssembly) per worker.
// The page sends {id, type, ...}; each reply is {id, ok, ...}. "depth" also
// posts {id, progress: {done, total}} while it runs. Cancelling = terminate().
// app.js usually starts workers from a blob holding mb_solver.js + this file, so
// they need no network; loaded on its own, the worker fetches the solver.
if (typeof createMbSolver === 'undefined') importScripts('mb_solver.js');

let M = null;
const ready = createMbSolver().then((m) => { M = m; });

function solutionsText() {
  return M.UTF8ToString(M._mb_solutions_text());
}

const handlers = {
  setup({ start, goal, fat }) {
    const p = M._malloc(72);
    M.HEAPU8.set(start, p);
    M.HEAPU8.set(goal, p + 36);
    const err = M.UTF8ToString(M._mb_setup(p, p + 36, fat ? 1 : 0,
      fat ? fat.start[0] : 0, fat ? fat.start[1] : 0, fat ? fat.goal[0] : 0, fat ? fat.goal[1] : 0));
    M._free(p);
    if (err) throw new Error(err);
    return {};
  },

  small({ maxDepth }) {
    const depth = M._mb_solve_small(maxDepth);
    if (depth < 0) return { depth };
    return { depth, count: M._mb_solution_count(), invalid: M._mb_invalid_count(), text: solutionsText() };
  },

  goal4() {
    M._mb_build_goal4();
    return { size: M._mb_layer_size(1, 4), memory: M._mb_memory_bytes(), heap: M.HEAPU8.length };
  },

  start4({ part, parts }) {
    M._mb_build_start4(part, parts);
    return { size: M._mb_layer_size(0, 4), memory: M._mb_memory_bytes(), heap: M.HEAPU8.length };
  },

  depth({ depth }, id) {
    const total = M._mb_begin_depth(depth);
    let done = 0;
    let chunk = 256;
    let lastPost = 0;
    for (;;) {
      const t0 = performance.now();
      const finished = M._mb_step(chunk);
      done = Math.min(total, done + chunk);
      const now = performance.now();
      // aim for ~40 ms per step so progress stays live
      const took = Math.max(0.1, now - t0);
      chunk = Math.max(16, Math.min(1 << 20, Math.round(chunk * Math.min(4, 40 / took))));
      if (finished) break;
      if (now - lastPost > 120) {
        lastPost = now;
        postMessage({ id, progress: { done, total, streamed: M._mb_streamed() } });
      }
    }
    M._mb_finish_depth();
    const count = M._mb_solution_count();
    return {
      count, total, streamed: M._mb_streamed(), invalid: M._mb_invalid_count(),
      text: count > 0 ? solutionsText() : '',
    };
  },
};

onmessage = async (e) => {
  await ready;
  const { id, type } = e.data;
  try {
    postMessage({ id, ok: true, ...handlers[type](e.data, id) });
  } catch (err) {
    postMessage({ id, ok: false, error: String(err && err.message || err) });
  }
};
