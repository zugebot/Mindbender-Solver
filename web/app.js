// web/app.js - board editor, the solver workers, results and replay.
'use strict';

(() => {
  const COLORS = ['Red', 'Green', 'Blue', 'Orange', 'Yellow', 'Purple', 'White', 'Cyan'];
  const MAX_DEPTH = 10;
  const LIST_PAGE = 300;
  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- model
  const blank = () => ({ cells: Array(36).fill(6), fat: null });
  const state = {
    start: blank(),
    goal: blank(),
    color: 0,
    moveTool: false,
    undo: [],
    preset: '',
  };

  const cloneBoard = (b) => ({ cells: b.cells.slice(), fat: b.fat ? { ...b.fat } : null });
  const sameBoard = (a, b) => a.cells.every((v, i) => v === b.cells[i])
    && JSON.stringify(a.fat) === JSON.stringify(b.fat);

  function pushUndo() {
    state.undo.push({ start: cloneBoard(state.start), goal: cloneBoard(state.goal), preset: state.preset });
    if (state.undo.length > 200) state.undo.shift();
  }

  const pieceCells = (f) => [f.y * 6 + f.x, f.y * 6 + f.x + 1, (f.y + 1) * 6 + f.x, (f.y + 1) * 6 + f.x + 1];
  const inPiece = (f, r, c) => !!f && r >= f.y && r <= f.y + 1 && c >= f.x && c <= f.x + 1;

  function paint(board, r, c, color) {
    if (inPiece(board.fat, r, c)) {
      for (const i of pieceCells(board.fat)) board.cells[i] = color;
    } else {
      board.cells[r * 6 + c] = color;
    }
  }

  // The piece's cells take its colour at the new place; the cells it leaves keep theirs.
  function placePiece(board, r, c) {
    const x = Math.min(c, 4), y = Math.min(r, 4);
    const color = board.fat ? board.cells[board.fat.y * 6 + board.fat.x] : board.cells[y * 6 + x];
    board.fat = { x, y };
    for (const i of pieceCells(board.fat)) board.cells[i] = color;
  }

  // ---------------------------------------------------------------- moves (for replay)
  // "R14" row 1 right 4, "C32" column 3 down 2, "R453" / "C231" the 2x2 piece's two lanes.
  function parseMove(name) {
    const d = name.slice(1);
    const lanes = d.length === 2 ? [+d[0] - 1] : [+d[0] - 1, +d[1] - 1];
    return { col: name[0] === 'C', lanes, n: +d[d.length - 1] };
  }

  function applyMove(board, mv) {
    const out = cloneBoard(board);
    for (const lane of mv.lanes) {
      for (let i = 0; i < 6; i++) {
        const j = (i + mv.n) % 6;
        if (mv.col) out.cells[j * 6 + lane] = board.cells[i * 6 + lane];
        else out.cells[lane * 6 + j] = board.cells[lane * 6 + i];
      }
    }
    if (out.fat && mv.lanes.length === 2) {
      if (mv.col) out.fat.y = (out.fat.y + mv.n) % 6;
      else out.fat.x = (out.fat.x + mv.n) % 6;
    }
    return out;
  }

  function laneCells(mv) {
    const s = new Set();
    for (const lane of mv.lanes) {
      for (let i = 0; i < 6; i++) s.add(mv.col ? i * 6 + lane : lane * 6 + i);
    }
    return s;
  }

  function describeMove(mv) {
    const what = mv.lanes.length === 2
      ? `${mv.col ? 'columns' : 'rows'} ${mv.lanes[0] + 1} and ${mv.lanes[1] + 1} (the 2x2 piece)`
      : `${mv.col ? 'column' : 'row'} ${mv.lanes[0] + 1}`;
    return `${what} ${mv.col ? 'down' : 'right'} ${mv.n}`;
  }

  // ---------------------------------------------------------------- rendering
  function renderBoard(el, board, highlight) {
    const frag = document.createDocumentFragment();
    const f = board.fat;
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const piece = inPiece(f, r, c);
        if (piece && (r !== f.y || c !== f.x)) continue;
        const color = board.cells[r * 6 + c];
        const d = document.createElement('div');
        d.className = piece ? 'cell piece' : 'cell';
        d.dataset.r = r;
        d.dataset.c = c;
        d.style.setProperty('--col', `var(--c${color})`);
        d.style.gridRow = `${r + 1} / span ${piece ? 2 : 1}`;
        d.style.gridColumn = `${c + 1} / span ${piece ? 2 : 1}`;
        d.setAttribute('role', 'gridcell');
        d.setAttribute('aria-label', `${piece ? '2x2 piece, ' : ''}row ${r + 1}, column ${c + 1}: ${COLORS[color]}`);
        if (highlight) {
          const idx = piece ? pieceCells(f) : [r * 6 + c];
          if (idx.some((i) => highlight.has(i))) d.classList.add('lane-hl');
        }
        frag.appendChild(d);
      }
    }
    el.replaceChildren(frag);
  }

  function renderBoards() {
    renderBoard($('startBoard'), state.start);
    renderBoard($('goalBoard'), state.goal);
    $('startBoard').classList.toggle('moving', state.moveTool && !!state.start.fat);
    $('goalBoard').classList.toggle('moving', state.moveTool && !!state.goal.fat);
  }

  function colourCounts(board) {
    const n = Array(8).fill(0);
    for (const v of board.cells) n[v]++;
    return n;
  }

  function renderCounts() {
    const a = colourCounts(state.start), b = colourCounts(state.goal);
    const el = $('counts');
    el.replaceChildren();
    for (let k = 0; k < 8; k++) {
      if (a[k] === 0 && b[k] === 0) continue;
      const chip = document.createElement('span');
      chip.className = a[k] === b[k] ? 'count' : 'count bad';
      chip.title = `${COLORS[k]}: ${a[k]} on the start board, ${b[k]} on the goal board`;
      const dot = document.createElement('i');
      dot.style.setProperty('--col', `var(--c${k})`);
      chip.append(dot, a[k] === b[k] ? `${a[k]}` : `${a[k]} / ${b[k]}`);
      el.appendChild(chip);
    }
  }

  function renderPalette() {
    const el = $('palette');
    el.replaceChildren();
    COLORS.forEach((name, k) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'swatch';
      b.style.setProperty('--col', `var(--c${k})`);
      b.setAttribute('aria-pressed', String(!state.moveTool && state.color === k));
      b.setAttribute('aria-label', name);
      b.title = `${name} (${k + 1})`;
      const s = document.createElement('span');
      s.textContent = k + 1;
      b.appendChild(s);
      b.addEventListener('click', () => selectColour(k));
      el.appendChild(b);
    });
    const hasFat = !!state.start.fat;
    $('moveTool').hidden = !hasFat;
    $('moveTool').setAttribute('aria-pressed', String(state.moveTool && hasFat));
  }

  function selectColour(k) {
    state.color = k;
    state.moveTool = false;
    renderPalette();
    renderBoards();
  }

  // ---------------------------------------------------------------- editing
  function attachEditor(el, which) {
    let painting = false;
    const cellAt = (e) => {
      const t = document.elementFromPoint(e.clientX, e.clientY);
      const cell = t && t.closest('.cell');
      return cell && el.contains(cell) ? cell : null;
    };
    const apply = (cell) => {
      const r = +cell.dataset.r, c = +cell.dataset.c;
      const b = state[which];
      const before = JSON.stringify(b);
      if (state.moveTool && b.fat) placePiece(b, r, c);
      else paint(b, r, c, state.color);
      if (JSON.stringify(b) !== before) renderBoard(el, b);
    };
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      const cell = cellAt(e);
      if (!cell) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      pushUndo();
      painting = true;
      apply(cell);
    });
    el.addEventListener('pointermove', (e) => {
      if (!painting) return;
      const cell = cellAt(e);
      if (cell) apply(cell);
    });
    const stop = () => {
      if (!painting) return;
      painting = false;
      changed();
    };
    el.addEventListener('pointerup', stop);
    el.addEventListener('pointercancel', stop);
    el.addEventListener('contextmenu', (e) => {
      const cell = cellAt(e);
      if (!cell) return;
      e.preventDefault();
      selectColour(state[which].cells[+cell.dataset.r * 6 + +cell.dataset.c]);
    });
  }

  function changed({ keepPreset = false } = {}) {
    if (!keepPreset && state.preset) {
      const p = presetByName(state.preset);
      if (!p || !sameBoard(state.start, presetBoard(p, 'start')) || !sameBoard(state.goal, presetBoard(p, 'goal'))) {
        state.preset = '';
      }
    }
    $('preset').value = state.preset;
    $('fatToggle').checked = !!state.start.fat;
    renderBoards();
    renderPalette();
    renderCounts();
    renderPresetNote();
    writeHash();
    markStale();
    if (!run) showMessage('');
  }

  function setFat(on) {
    pushUndo();
    for (const b of [state.start, state.goal]) {
      if (on && !b.fat) placePiece(b, 2, 2);
      if (!on) b.fat = null;
    }
    if (!on) state.moveTool = false;
    changed();
  }

  // ---------------------------------------------------------------- presets and links
  const PUZZLES = self.MB_PUZZLES || [];
  const presetByName = (name) => PUZZLES.find((p) => p.name === name);
  const presetBoard = (p, side) => ({
    cells: p[side].slice(),
    fat: p.fat ? { x: p.fat[side][0], y: p.fat[side][1] } : null,
  });

  function fillPresets() {
    const sel = $('preset');
    let group = null, world = '';
    for (const p of PUZZLES) {
      const w = p.name.split('-')[0];
      if (w !== world) {
        world = w;
        group = document.createElement('optgroup');
        group.label = `World ${w}`;
        sel.appendChild(group);
      }
      const o = document.createElement('option');
      o.value = p.name;
      const len = p.moves ? `${p.moves} moves` : `${p.atLeast}+ moves`;
      o.textContent = `${p.name}  ·  ${len}${p.fat ? '  ·  2x2' : ''}`;
      group.appendChild(o);
    }
  }

  function loadPreset(name) {
    const p = presetByName(name);
    if (!p) return;
    pushUndo();
    state.start = presetBoard(p, 'start');
    state.goal = presetBoard(p, 'goal');
    state.preset = name;
    state.moveTool = false;
    changed({ keepPreset: true });
  }

  function renderPresetNote() {
    const el = $('presetNote');
    const p = state.preset && presetByName(state.preset);
    el.hidden = !p;
    el.classList.remove('warn');
    if (!p) return;
    if (p.moves && p.moves <= MAX_DEPTH) {
      el.textContent = `Puzzle ${p.name}: the shortest solution is ${p.moves} moves (${p.solutions.toLocaleString()} of them).`;
    } else if (p.moves) {
      el.textContent = `Puzzle ${p.name} needs ${p.moves} moves, more than this page searches (${MAX_DEPTH}). The desktop solver found its ${p.solutions.toLocaleString()} solutions.`;
      el.classList.add('warn');
    } else {
      el.textContent = `Puzzle ${p.name} is not solved yet: it needs at least ${p.atLeast} moves, far beyond what a browser can search.`;
      el.classList.add('warn');
    }
  }

  function writeHash() {
    const enc = (b) => b.cells.join('');
    let h = `s=${enc(state.start)}&g=${enc(state.goal)}`;
    if (state.start.fat) {
      const a = state.start.fat, b = state.goal.fat;
      h += `&f=${a.x}${a.y}${b.x}${b.y}`;
    }
    if (state.preset) h = `p=${state.preset}&` + h;
    history.replaceState(null, '', `#${h}`);
  }

  function readHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    const s = q.get('s'), g = q.get('g'), f = q.get('f');
    const ok = (t) => t && /^[0-7]{36}$/.test(t);
    if (ok(s) && ok(g)) {
      state.start = { cells: [...s].map(Number), fat: null };
      state.goal = { cells: [...g].map(Number), fat: null };
      if (f && /^[0-4]{4}$/.test(f)) {
        state.start.fat = { x: +f[0], y: +f[1] };
        state.goal.fat = { x: +f[2], y: +f[3] };
      }
      state.preset = q.get('p') || '';
      return true;
    }
    if (q.get('p') && presetByName(q.get('p'))) {
      const p = presetByName(q.get('p'));
      state.start = presetBoard(p, 'start');
      state.goal = presetBoard(p, 'goal');
      state.preset = p.name;
      return true;
    }
    return false;
  }

  // ---------------------------------------------------------------- solver workers
  class Cancelled extends Error {}

  // Workers start from one blob of mb_solver.js + solver-worker.js, fetched once,
  // so starting 16 of them costs no network. One worker is kept warm for the next solve.
  let workerUrl = 'solver-worker.js';
  const workerUrlReady = Promise.all(['mb_solver.js', 'solver-worker.js'].map((f) =>
    fetch(f).then((r) => {
      if (!r.ok) throw new Error(f);
      return r.text();
    })))
    .then(([solver, worker]) => {
      workerUrl = URL.createObjectURL(new Blob([solver, '\n', worker], { type: 'text/javascript' }));
    })
    .catch(() => {});
  let warm = null;
  const warmUp = () => workerUrlReady.then(() => { if (!warm && !run) warm = new SolverWorker(); });

  class SolverWorker {
    constructor() {
      this.w = new Worker(workerUrl);
      this.next = 1;
      this.pending = new Map();
      this.w.onmessage = (e) => {
        const m = e.data;
        const p = this.pending.get(m.id);
        if (!p) return;
        if (m.progress) {
          if (p.onProgress) p.onProgress(m.progress);
          return;
        }
        this.pending.delete(m.id);
        if (m.ok) p.resolve(m);
        else p.reject(new Error(m.error));
      };
      this.w.onerror = (e) => {
        e.preventDefault();
        this.failAll(new Error(e.message || 'the solver stopped (out of memory?)'));
      };
    }

    call(type, data = {}, onProgress = null) {
      const id = this.next++;
      return new Promise((resolve, reject) => {
        this.pending.set(id, { resolve, reject, onProgress });
        this.w.postMessage({ id, type, ...data });
      });
    }

    failAll(err) {
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    }

    terminate() {
      this.w.terminate();
      this.failAll(new Cancelled('cancelled'));
    }
  }

  const hardwareThreads = Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 32));

  // RAM the workers may use together (each holds the 4-move goal table).
  function memoryBudget() {
    const dm = navigator.deviceMemory;
    if (dm && dm <= 2) return 400e6;
    if (dm && dm <= 4) return 800e6;
    return 1600e6;
  }

  let run = null;
  let result = null;

  function setStatus(text, fraction) {
    $('statusText').textContent = text;
    const fill = $('progressFill');
    if (fraction == null) {
      fill.classList.add('indet');
      fill.style.width = '';
    } else {
      fill.classList.remove('indet');
      fill.style.width = `${Math.round(fraction * 1000) / 10}%`;
    }
  }

  const fmtTime = (s) => (s < 10 ? `${s.toFixed(1)} s` : s < 90 ? `${Math.round(s)} s`
    : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`);

  function showMessage(text, isError) {
    const el = $('message');
    el.hidden = !text;
    el.textContent = text || '';
    el.classList.toggle('error', !!isError);
  }

  function problemSpec() {
    const a = colourCounts(state.start), b = colourCounts(state.goal);
    const diff = [];
    for (let k = 0; k < 8; k++) {
      if (a[k] !== b[k]) diff.push(`${COLORS[k].toLowerCase()} ${a[k]} vs ${b[k]}`);
    }
    if (diff.length) {
      return { error: `The boards must have the same colours. Start vs goal: ${diff.join(', ')}.` };
    }
    const fat = state.start.fat
      ? { start: [state.start.fat.x, state.start.fat.y], goal: [state.goal.fat.x, state.goal.fat.y] } : null;
    return { spec: { start: state.start.cells.slice(), goal: state.goal.cells.slice(), fat } };
  }

  async function solve() {
    if (run) return;
    const { spec, error } = problemSpec();
    if (error) {
      showMessage(error, true);
      return;
    }
    showMessage('');
    const maxDepth = +$('maxDepth').value;
    const threads = +$('threads').value || hardwareThreads;
    run = { workers: [], cancelled: false, t0: performance.now(), boards: { start: cloneBoard(state.start), goal: cloneBoard(state.goal) } };
    const thisRun = run;
    $('solveBtn').hidden = true;
    $('cancelBtn').hidden = false;
    $('progress').hidden = false;
    const tick = setInterval(() => {
      $('statusTime').textContent = fmtTime((performance.now() - thisRun.t0) / 1000);
    }, 100);
    try {
      await workerUrlReady;
      const w0 = warm || new SolverWorker();
      warm = null;
      run.workers.push(w0);
      await w0.call('setup', spec);
      setStatus(`Checking 0 to ${Math.min(6, maxDepth)} moves...`, null);
      const small = await w0.call('small', { maxDepth });
      if (small.depth >= 0) {
        finish(thisRun, { depth: small.depth, count: small.count, text: small.text, threads: 1, maxDepth });
        return;
      }
      if (maxDepth <= 6) {
        finish(thisRun, { depth: -1, threads: 1, maxDepth });
        return;
      }
      setStatus('Building the table of boards 4 moves from the goal...', null);
      const g = await w0.call('goal4');
      const perWorker = g.memory * 1.5 + 32e6;
      const n = Math.max(1, Math.min(threads, Math.floor(memoryBudget() / perWorker)));
      for (let i = 1; i < n; i++) run.workers.push(new SolverWorker());
      setStatus(`Building tables on ${n} thread${n > 1 ? 's' : ''}...`, null);
      await Promise.all(run.workers.slice(1).map((w) => w.call('setup', spec).then(() => w.call('goal4'))));
      await Promise.all(run.workers.map((w, i) => w.call('start4', { part: i, parts: n })));
      for (let d = 7; d <= maxDepth; d++) {
        const prog = run.workers.map(() => ({ done: 0, total: 1 }));
        const tStart = performance.now();
        const update = () => {
          const done = prog.reduce((s, p) => s + p.done, 0);
          const total = prog.reduce((s, p) => s + p.total, 0);
          const f = total ? done / total : 0;
          const el = (performance.now() - tStart) / 1000;
          const eta = f > 0.03 && el > 1.5 ? ` · about ${fmtTime(el * (1 - f) / f)} left` : '';
          setStatus(`Checking ${d} moves... ${Math.floor(f * 100)}%${eta}`, f);
        };
        setStatus(`Checking ${d} moves...`, 0);
        const res = await Promise.all(run.workers.map((w, i) =>
          w.call('depth', { depth: d }, (p) => { prog[i] = p; update(); })));
        const count = res.reduce((s, r) => s + r.count, 0);
        if (count > 0) {
          finish(thisRun, { depth: d, count, text: res.map((r) => r.text).join(''), threads: n, maxDepth });
          return;
        }
      }
      finish(thisRun, { depth: -1, threads: n, maxDepth });
    } catch (err) {
      if (!thisRun.cancelled) {
        let msg = `The solver failed: ${err.message}`;
        if (location.protocol === 'file:') {
          msg = 'Browsers block workers on file:// pages. Serve this folder instead, e.g. "python -m http.server" in web/, then open http://localhost:8000/.';
        }
        showMessage(msg, true);
      }
    } finally {
      clearInterval(tick);
      for (const w of thisRun.workers) w.terminate();
      if (run === thisRun) run = null;
      $('solveBtn').hidden = false;
      $('cancelBtn').hidden = true;
      $('progress').hidden = true;
      warmUp();
    }
  }

  function cancel() {
    if (!run) return;
    run.cancelled = true;
    for (const w of run.workers) w.terminate();
    showMessage('Cancelled.');
  }

  // ---------------------------------------------------------------- results
  function finish(r, res) {
    if (r.cancelled) return;
    const secs = (performance.now() - r.t0) / 1000;
    const lines = res.text ? res.text.split('\n').filter(Boolean).sort() : [];
    result = { ...res, lines, secs, boards: r.boards, shown: 0, selected: -1, step: 0, playing: null };
    $('results').hidden = false;
    const threads = `${res.threads} thread${res.threads > 1 ? 's' : ''}`;
    if (res.depth < 0) {
      $('resultTitle').textContent = `No solution in ${res.maxDepth} moves or fewer`;
      $('resultSub').textContent = `Every move sequence up to ${res.maxDepth} moves was checked (${fmtTime(secs)}, ${threads}).`;
    } else if (res.depth === 0) {
      $('resultTitle').textContent = 'Already solved';
      $('resultSub').textContent = 'The start board is the goal board.';
    } else {
      const n = res.count;
      $('resultTitle').textContent = `${res.depth} move${res.depth > 1 ? 's' : ''}`;
      let sub = `${n.toLocaleString()} shortest solution${n > 1 ? 's' : ''}, none shorter · ${fmtTime(secs)} on ${threads}`;
      if (lines.length < n) sub += ` · ${lines.length.toLocaleString()} listed`;
      $('resultSub').textContent = sub;
    }
    $('copyBtn').hidden = $('downloadBtn').hidden = lines.length === 0;
    $('solutionList').replaceChildren();
    document.querySelector('.results-body').hidden = lines.length === 0;
    showMore();
    if (lines.length) selectSolution(0);
    $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function markStale() {
    if (!result || $('results').hidden) return;
    const same = sameBoard(result.boards.start, state.start) && sameBoard(result.boards.goal, state.goal);
    $('results').classList.toggle('stale', !same);
    const note = ' · the boards have changed since';
    const sub = $('resultSub');
    if (!same && !sub.textContent.endsWith(note)) sub.textContent += note;
    if (same && sub.textContent.endsWith(note)) sub.textContent = sub.textContent.slice(0, -note.length);
  }

  function showMore() {
    const list = $('solutionList');
    const end = Math.min(result.lines.length, result.shown + LIST_PAGE);
    const frag = document.createDocumentFragment();
    for (let i = result.shown; i < end; i++) {
      const li = document.createElement('li');
      li.textContent = result.lines[i];
      li.dataset.i = i;
      li.setAttribute('aria-selected', 'false');
      frag.appendChild(li);
    }
    list.appendChild(frag);
    result.shown = end;
    const left = result.lines.length - end;
    $('moreBtn').hidden = left <= 0;
    $('moreBtn').textContent = `Show more (${left.toLocaleString()} left)`;
  }

  function selectSolution(i) {
    stopPlay();
    const prev = $('solutionList').querySelector('[aria-selected="true"]');
    if (prev) prev.setAttribute('aria-selected', 'false');
    const li = $('solutionList').querySelector(`[data-i="${i}"]`);
    if (li) li.setAttribute('aria-selected', 'true');
    const moves = result.lines[i].split(' ').map(parseMove);
    const boards = [cloneBoard(result.boards.start)];
    for (const mv of moves) boards.push(applyMove(boards[boards.length - 1], mv));
    result.selected = i;
    result.moves = moves;
    result.names = result.lines[i].split(' ');
    result.replay = boards;
    const chips = $('replayChips');
    chips.replaceChildren();
    result.names.forEach((name, k) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = name;
      b.title = describeMove(moves[k]);
      b.addEventListener('click', () => showStep(k));
      chips.appendChild(b);
    });
    showStep(0);
  }

  function showStep(k) {
    if (!result || !result.replay) return;
    const n = result.moves.length;
    k = Math.max(0, Math.min(n, k));
    result.step = k;
    const next = k < n ? result.moves[k] : null;
    renderBoard($('replayBoard'), result.replay[k], next ? laneCells(next) : null);
    $('replayBoard').classList.add('readonly');
    $('stepText').textContent = next
      ? `Move ${k + 1} of ${n}: ${describeMove(next)}`
      : 'Solved!';
    [...$('replayChips').children].forEach((c, i) => {
      c.classList.toggle('done', i < k);
      c.classList.toggle('now', i === k);
    });
  }

  function stopPlay() {
    if (result && result.playing) {
      clearInterval(result.playing);
      result.playing = null;
      $('playBtn').innerHTML = '&#x25B6;';
      $('playBtn').setAttribute('aria-label', 'Play');
    }
  }

  function togglePlay() {
    if (!result || !result.replay) return;
    if (result.playing) {
      stopPlay();
      return;
    }
    if (result.step >= result.moves.length) showStep(0);
    $('playBtn').innerHTML = '&#x23F8;';
    $('playBtn').setAttribute('aria-label', 'Pause');
    result.playing = setInterval(() => {
      if (result.step >= result.moves.length) {
        stopPlay();
        return;
      }
      showStep(result.step + 1);
    }, 750);
  }

  function download(name, text) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------------------------------------------------------------- wiring
  function init() {
    fillPresets();
    for (let d = 1; d <= MAX_DEPTH; d++) {
      const o = new Option(`${d} move${d > 1 ? 's' : ''}`, d);
      $('maxDepth').appendChild(o);
    }
    $('maxDepth').value = MAX_DEPTH;
    $('threads').appendChild(new Option(`Auto (${hardwareThreads})`, 0));
    for (const t of [1, 2, 4, 8, 12, 16, 24, 32]) {
      if (t < hardwareThreads) $('threads').appendChild(new Option(String(t), t));
    }
    $('engineNote').textContent = `Runs on up to ${hardwareThreads} threads with WebAssembly. Up to 8 moves takes about a second; 9 and 10 moves take seconds to a few minutes, depending on the puzzle and the device.`;

    attachEditor($('startBoard'), 'start');
    attachEditor($('goalBoard'), 'goal');

    $('preset').addEventListener('change', (e) => {
      if (e.target.value) loadPreset(e.target.value);
      else { state.preset = ''; changed(); }
    });
    $('fatToggle').addEventListener('change', (e) => setFat(e.target.checked));
    $('moveTool').addEventListener('click', () => {
      state.moveTool = !state.moveTool;
      renderPalette();
      renderBoards();
    });
    $('undoBtn').addEventListener('click', undo);
    $('swapBtn').addEventListener('click', () => {
      pushUndo();
      [state.start, state.goal] = [state.goal, state.start];
      changed();
    });
    $('shareBtn').addEventListener('click', async () => {
      writeHash();
      try {
        await navigator.clipboard.writeText(location.href);
        flash($('shareBtn'), 'Link copied');
      } catch {
        prompt('Copy this link:', location.href);
      }
    });
    document.querySelectorAll('.mini-actions .link').forEach((b) => b.addEventListener('click', () => {
      const which = b.dataset.board, other = which === 'start' ? 'goal' : 'start';
      pushUndo();
      if (b.dataset.act === 'fill') {
        state[which].cells.fill(state.color);
      } else {
        state[which] = cloneBoard(state[other]);
      }
      changed();
    }));

    $('solveBtn').addEventListener('click', solve);
    $('cancelBtn').addEventListener('click', cancel);
    $('moreBtn').addEventListener('click', showMore);
    $('solutionList').addEventListener('click', (e) => {
      const li = e.target.closest('li');
      if (li) selectSolution(+li.dataset.i);
    });
    $('firstBtn').addEventListener('click', () => { stopPlay(); showStep(0); });
    $('prevBtn').addEventListener('click', () => { stopPlay(); showStep(result.step - 1); });
    $('nextBtn').addEventListener('click', () => { stopPlay(); showStep(result.step + 1); });
    $('lastBtn').addEventListener('click', () => { stopPlay(); showStep(Infinity); });
    $('playBtn').addEventListener('click', togglePlay);
    $('copyBtn').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(result.lines.join('\n') + '\n');
        flash($('copyBtn'), 'Copied');
      } catch { /* clipboard blocked */ }
    });
    $('downloadBtn').addEventListener('click', () => {
      const name = (state.preset || 'puzzle') + `_${result.depth}moves_${result.count}.txt`;
      download(name, result.lines.join('\n') + '\n');
    });

    document.addEventListener('keydown', (e) => {
      if (e.target.closest('select, input, textarea')) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undo();
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey && /^[1-8]$/.test(e.key)) {
        selectColour(+e.key - 1);
      } else if (e.key.toLowerCase() === 'm' && state.start.fat) {
        state.moveTool = !state.moveTool;
        renderPalette();
        renderBoards();
      } else if (result && result.replay && !$('results').hidden && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) {
        e.preventDefault();
        stopPlay();
        showStep(result.step + (e.key === 'ArrowRight' ? 1 : -1));
      }
    });

    if (!readHash()) {
      const p = presetByName('4-3');
      if (p) {
        state.start = presetBoard(p, 'start');
        state.goal = presetBoard(p, 'goal');
        state.preset = p.name;
      }
    }
    changed({ keepPreset: true });
    warmUp();
  }

  function undo() {
    const s = state.undo.pop();
    if (!s) return;
    state.start = s.start;
    state.goal = s.goal;
    state.preset = s.preset;
    changed({ keepPreset: true });
  }

  function flash(btn, text) {
    const old = btn.textContent;
    btn.textContent = text;
    setTimeout(() => { btn.textContent = old; }, 1400);
  }

  init();
})();
