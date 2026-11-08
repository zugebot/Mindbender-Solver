// Runs the browser solver (web/mb_solver.js, the WebAssembly build) on game puzzles in Node,
// the way the page does: setup, the small search, the 4-move goal layer, then each depth from
// 7 with the seeds split into --parts parts. The parts run one after another on one thread
// (each with its own copy of the solver, as each page worker has), so the time is what one
// browser thread takes for the whole search; the page divides the depth part of it among its
// workers. Checks the solution count against the known answer.
//   node web/tools/wasm_bench.mjs 9-2,11-1 [--max 12] [--parts 1]
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';
import vm from 'node:vm';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name, def) => {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : def;
};
const names = (process.argv[2] || '9-2').split(',');
const maxDepth = +arg('--max', 12);
const parts = +arg('--parts', 1);

const ctx = { self: {} };
vm.runInNewContext(readFileSync(join(web, 'puzzles.js'), 'utf8'), ctx);
const puzzles = ctx.self.MB_PUZZLES;
// mb_solver.js is a classic script (MODULARIZE, not ES6) that exports createMbSolver under Node
const createMbSolver = createRequire(import.meta.url)(join(web, 'mb_solver.js'));

async function solver(p) {
  const M = await createMbSolver();
  const buf = M._malloc(72);
  M.HEAPU8.set(p.start, buf);
  M.HEAPU8.set(p.goal, buf + 36);
  const err = M.UTF8ToString(M._mb_setup(buf, buf + 36, p.fat ? 1 : 0,
    p.fat ? p.fat.start[0] : 0, p.fat ? p.fat.start[1] : 0, p.fat ? p.fat.goal[0] : 0, p.fat ? p.fat.goal[1] : 0));
  M._free(buf);
  if (err) throw new Error(err);
  return M;
}

for (const name of names) {
  const p = puzzles.find((q) => q.name === name);
  if (!p) {
    console.log(`${name}: no such puzzle`);
    continue;
  }
  const t0 = performance.now();
  const first = await solver(p);
  let depth = first._mb_solve_small(maxDepth), count = depth >= 0 ? first._mb_solution_count() : 0;
  let goalMs = 0, startMs = 0, streamMs = 0, streamed = 0, memory = 0;
  if (depth < 0 && maxDepth >= 7) {
    const ws = [];
    for (let i = 0; i < parts; i++) {
      const M = i === 0 ? first : await solver(p);
      let t = performance.now();
      M._mb_build_goal4();
      goalMs += performance.now() - t;
      t = performance.now();
      M._mb_build_start4(i, parts);
      startMs += performance.now() - t;
      memory = Math.max(memory, M._mb_memory_bytes());
      ws.push(M);
    }
    for (let d = 7; d <= maxDepth && depth < 0; d++) {
      let c = 0;
      const t = performance.now();
      for (const M of ws) {
        M._mb_begin_depth(d);
        while (!M._mb_step(1 << 16)) { /* the whole part */ }
        M._mb_finish_depth();
        c += M._mb_solution_count();
        streamed += M._mb_streamed();
      }
      streamMs += performance.now() - t;
      if (c > 0) {
        depth = d;
        count = c;
      }
    }
  }
  const want = p.moves ? `${p.moves} moves, ${p.solutions} solutions` : `unsolved, at least ${p.atLeast}`;
  const ok = depth === p.moves && count === p.solutions;
  console.log(`${name.padEnd(5)} ${p.fat ? '2x2' : '   '} ${ok ? 'ok  ' : 'DIFF'} found ${depth} moves, ${count} solutions (want ${want})`
    + `  total ${((performance.now() - t0) / 1000).toFixed(2)} s: goal side ${(goalMs / 1000).toFixed(2)} s,`
    + ` start side ${(startMs / 1000).toFixed(2)} s, stream ${(streamMs / 1000).toFixed(2)} s`
    + `  streamed ${streamed.toExponential(2)}  memory ${(memory / 1e6).toFixed(0)} MB a part`);
}
