// web/app.js - board editor, the solver workers, results and replay.
'use strict';

(() => {
  const COLORS = ['Red', 'Green', 'Blue', 'Orange', 'Yellow', 'Purple', 'White', 'Cyan'];
  const MAX_DEPTH = 10;
  const LIST_PAGE = 300;
  const $ = (id) => document.getElementById(id);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

  const ICONS = {
    help: '<circle cx="12" cy="12" r="9.5"/><path d="M9.2 9.2a2.9 2.9 0 0 1 5.6.9c0 1.9-2.8 2.7-2.8 2.7"/><path d="M12 17h.01"/>',
    undo: '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/>',
    bucket: '<path d="m19 11-8-8-8.6 8.6a2 2 0 0 0 0 2.8l5.2 5.2a2 2 0 0 0 2.8 0Z"/><path d="m5 2 5 5"/><path d="M2 13h15"/><path d="M22 20a2 2 0 1 1-4 0c0-1.6 1.7-2.4 2-4 .3 1.6 2 2.4 2 4Z"/>',
    copy: '<rect x="9" y="9" width="12" height="12" rx="2.5"/><path d="M5 15H4.5A1.5 1.5 0 0 1 3 13.5v-9A1.5 1.5 0 0 1 4.5 3h9A1.5 1.5 0 0 1 15 4.5V5"/>',
    swap: '<path d="m16 3 4 4-4 4"/><path d="M20 7H4"/><path d="m8 21-4-4 4-4"/><path d="M4 17h16"/>',
    piece: '<rect x="3" y="3" width="11" height="11" rx="3"/><circle cx="18.5" cy="8.5" r="2.5"/><circle cx="8.5" cy="18.5" r="2.5"/><circle cx="18.5" cy="18.5" r="2.5"/>',
    move: '<path d="M12 3v18M3 12h18"/><path d="m9 6 3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
    link: '<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>',
    left: '<path d="m15 18-6-6 6-6"/>',
    right: '<path d="m9 18 6-6-6-6"/>',
    first: '<path d="m17 18-6-6 6-6"/><path d="M7 6v12"/>',
    last: '<path d="m7 18 6-6-6-6"/><path d="M17 6v12"/>',
    play: '<path d="M8 5.5v13a1 1 0 0 0 1.5.9l10.2-6.5a1 1 0 0 0 0-1.7L9.5 4.6A1 1 0 0 0 8 5.5Z" fill="currentColor" stroke="none"/>',
    pause: '<rect x="6.5" y="5" width="4" height="14" rx="1.2" fill="currentColor" stroke="none"/><rect x="13.5" y="5" width="4" height="14" rx="1.2" fill="currentColor" stroke="none"/>',
    x: '<path d="M18 6 6 18M6 6l12 12"/>',
    pencil: '<path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    clipboard: '<rect x="8" y="2.5" width="8" height="4" rx="1"/><path d="M16 4.5h2a2 2 0 0 1 2 2V20a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6.5a2 2 0 0 1 2-2h2"/>',
    download: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
    check: '<path d="M20 6 9 17l-5-5"/>',
  };
  const icon = (name) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
  function setIcon(el, name) {
    const old = el.querySelector(':scope > svg.i');
    if (old) old.remove();
    el.insertAdjacentHTML('afterbegin', icon(name));
  }

  // ---------------------------------------------------------------- model
  const blank = () => ({ cells: Array(36).fill(6), fat: null });
  const state = {
    start: blank(),
    goal: blank(),
    color: 2,
    moveTool: false,
    undo: [],
    preset: '',
    side: 'start',          // the board being edited (the one shown on phones)
    mode: 'edit',           // 'edit' | 'replay'
  };

  // ids: which chuzzle sits in each cell (replay moves them, so each keeps its eyes)
  const cloneBoard = (b) => ({ cells: b.cells.slice(), fat: b.fat ? { ...b.fat } : null, ids: b.ids ? b.ids.slice() : undefined });
  const sameBoard = (a, b) => a.cells.every((v, i) => v === b.cells[i])
    && JSON.stringify(a.fat) === JSON.stringify(b.fat);
  const otherSide = (s) => (s === 'start' ? 'goal' : 'start');

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

  // ---------------------------------------------------------------- moves (replay)
  // "R14" row 1 right 4, "C32" column 3 down 2, "R453" / "C231" the 2x2 piece's two lanes.
  function parseMove(name) {
    const d = name.slice(1);
    const lanes = d.length === 2 ? [+d[0] - 1] : [+d[0] - 1, +d[1] - 1];
    return { col: name[0] === 'C', lanes, n: +d[d.length - 1], name };
  }

  function applyMove(board, mv) {
    const out = cloneBoard(board);
    const ids = board.ids || board.cells.map((_, i) => i);
    out.ids = ids.slice();
    for (const lane of mv.lanes) {
      for (let i = 0; i < 6; i++) {
        const j = (i + mv.n) % 6;
        const to = mv.col ? j * 6 + lane : lane * 6 + j, from = mv.col ? i * 6 + lane : lane * 6 + i;
        out.cells[to] = board.cells[from];
        out.ids[to] = ids[from];
      }
    }
    if (out.fat && mv.lanes.length === 2) {
      if (mv.col) out.fat.y = (out.fat.y + mv.n) % 6;
      else out.fat.x = (out.fat.x + mv.n) % 6;
    }
    return out;
  }

  function describeMove(mv) {
    const what = mv.lanes.length === 2
      ? `${mv.col ? 'columns' : 'rows'} ${mv.lanes[0] + 1}+${mv.lanes[1] + 1}`
      : `${mv.col ? 'column' : 'row'} ${mv.lanes[0] + 1}`;
    return `${what} ${mv.col ? 'down' : 'right'} ${mv.n}`;
  }

  // ---------------------------------------------------------------- board rendering
  // opts.band: highlight the lane(s) a move will shift. opts.anim: {mv, dir} slides
  // the lanes that just moved into place (dir -1 = the move was undone).
  function renderBoard(el, board, opts = {}) {
    const frag = document.createDocumentFragment();
    const f = board.fat;
    if (opts.band) {
      const mv = opts.band;
      const band = document.createElement('div');
      band.className = mv.col ? 'band col' : 'band';
      const lo = Math.min(...mv.lanes) + 1;
      const span = `${lo} / span ${mv.lanes.length}`;
      band.style.gridRow = mv.col ? '1 / -1' : span;
      band.style.gridColumn = mv.col ? span : '1 / -1';
      band.innerHTML = `<svg viewBox="0 0 10 10" aria-hidden="true"><path d="${mv.col ? 'M1 3h8L5 8z' : 'M3 1v8l5-4z'}"/></svg>`;
      frag.appendChild(band);
    }
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const piece = inPiece(f, r, c);
        if (piece && (r !== f.y || c !== f.x)) continue;
        const color = board.cells[r * 6 + c];
        const d = document.createElement('div');
        d.className = piece ? 'cell piece' : 'cell';
        d.dataset.r = r;
        d.dataset.c = c;
        setChuzzle(d, color, board.ids ? board.ids[r * 6 + c] : r * 6 + c);
        d.style.gridRow = `${r + 1} / span ${piece ? 2 : 1}`;
        d.style.gridColumn = `${c + 1} / span ${piece ? 2 : 1}`;
        d.setAttribute('role', 'gridcell');
        d.setAttribute('aria-label', `${piece ? '2x2 piece, ' : ''}row ${r + 1}, column ${c + 1}: ${COLORS[color]}`);
        frag.appendChild(d);
      }
    }
    el.replaceChildren(frag);
    if (opts.anim && !reducedMotion.matches) slideLanes(el, opts.anim.mv, opts.anim.dir);
  }

  // A chuzzle's sprite: colour row, and an eye shape, pupil colour and sway phase that
  // stay with it (like the game's random ones, but fixed per chuzzle).
  function setChuzzle(el, color, id) {
    el.style.setProperty('--k', color);
    el.style.setProperty('--es', (id * 7 + 3) % 4);
    el.style.setProperty('--ep', (id * 5 + 1) % 6);
    el.style.setProperty('--ph', ((id * 37) % 100) / 100);
  }

  // Each moved ball starts where it was and slides to its new place (in the game's
  // stretched frame for that direction); a ball that wrapped round the edge slides in
  // from outside while a copy slides out.
  function slideLanes(el, mv, dir) {
    const any = el.querySelector('.cell:not(.piece)') || el.querySelector('.cell');
    if (!any) return;
    const gap = parseFloat(getComputedStyle(el).columnGap) || 4;
    const pitch = (any.classList.contains('piece') ? (any.offsetWidth - gap) / 2 : any.offsetWidth) + gap;
    const d = dir * mv.n;
    const moving = [];
    for (const cell of el.querySelectorAll('.cell')) {
      const r = +cell.dataset.r, c = +cell.dataset.c;
      const piece = cell.classList.contains('piece');
      const lanes = mv.col ? (piece ? [c, c + 1] : [c]) : (piece ? [r, r + 1] : [r]);
      if (!lanes.some((l) => mv.lanes.includes(l))) continue;
      const j = mv.col ? r : c;
      const w = piece ? 2 : 1;
      moving.push([cell, -d, 0]);
      const wrapped = d > 0 ? j - d < 0 : j - d + w - 1 > 5;
      if (wrapped) {
        const ghost = cell.cloneNode(true);
        ghost.classList.add('ghost');
        ghost.removeAttribute('role');
        el.appendChild(ghost);
        moving.push([ghost, d > 0 ? 6 - d : -6 - d, d > 0 ? 6 : -6]);
      }
    }
    const tr = (k) => (mv.col ? `translateY(${k * pitch}px)` : `translateX(${k * pitch}px)`);
    const seg = mv.col ? (d > 0 ? 0 : 3) : (d > 0 ? 2 : 1);      // down, left, right, up
    for (const [node, from] of moving) {
      node.style.transition = 'none';
      node.style.transform = tr(from);
      node.style.setProperty('--f', (node.classList.contains('piece') ? 10 : 20) + seg);
      node.classList.add('mv');
    }
    el.getBoundingClientRect();
    const ms = 140 + 45 * mv.n;
    for (const [node, , to] of moving) {
      node.style.transition = `transform ${ms}ms cubic-bezier(.25,.8,.25,1)`;
      node.style.transform = to ? tr(to) : '';
    }
    setTimeout(() => {
      el.querySelectorAll('.ghost').forEach((g) => g.remove());
      for (const [node] of moving) node.classList.remove('mv');
    }, ms + 30);
  }

  function renderThumb(el, board) {
    const frag = document.createDocumentFragment();
    board.cells.forEach((v, k) => {
      const i = document.createElement('i');
      i.style.setProperty('--col', `var(--c${v})`);
      if (inPiece(board.fat, Math.floor(k / 6), k % 6)) i.className = 'p';
      frag.appendChild(i);
    });
    el.replaceChildren(frag);
  }

  const colourCounts = (board) => {
    const n = Array(8).fill(0);
    for (const v of board.cells) n[v]++;
    return n;
  };

  function renderPalette() {
    const a = colourCounts(state.start), b = colourCounts(state.goal);
    const el = $('palette');
    if (!el.children.length) {
      COLORS.forEach((name, k) => {
        const s = document.createElement('button');
        s.type = 'button';
        s.className = 'swatch';
        s.setAttribute('role', 'radio');
        setChuzzle(s, k, k * 3 + 1);
        s.innerHTML = '<span class="n"></span>';
        s.addEventListener('click', () => selectColour(k));
        el.appendChild(s);
      });
    }
    [...el.children].forEach((s, k) => {
      const ok = a[k] === b[k];
      s.setAttribute('aria-checked', String(!state.moveTool && state.color === k));
      s.setAttribute('aria-label', `${COLORS[k]}: ${a[k]} on start, ${b[k]} on goal`);
      s.title = `${COLORS[k]} (key ${k + 1})${ok ? '' : ` - ${a[k]} on start, ${b[k]} on goal`}`;
      const n = s.firstChild;
      n.className = ok ? 'n' : 'n bad';
      n.textContent = a[k] || b[k] ? (ok ? a[k] : `${a[k]}≠${b[k]}`) : '';
    });
    const diff = [];
    for (let k = 0; k < 8; k++) if (a[k] !== b[k]) diff.push(`${COLORS[k].toLowerCase()} ${a[k]}≠${b[k]}`);
    $('countWarn').hidden = diff.length === 0;
    $('countWarn').textContent = `Both boards need the same balls (start≠goal: ${diff.join(', ')})`;
  }

  function render() {
    const replay = state.mode === 'replay' && result && result.replay;
    for (const wrap of document.querySelectorAll('.board-wrap')) {
      wrap.classList.toggle('active', wrap.dataset.side === state.side);
    }
    if (!replay) renderBoard($('startBoard'), state.start);
    renderBoard($('goalBoard'), state.goal);
    for (const side of ['start', 'goal']) {
      const b = $(`${side}Board`);
      b.classList.toggle('moving', !replay && state.moveTool && !!state[side].fat);
      b.classList.toggle('readonly', side === 'start' && !!replay);
    }
    renderThumb($('thumbStart'), replay ? result.replay[result.step] : state.start);
    renderThumb($('thumbGoal'), state.goal);
    $('tabStart').setAttribute('aria-selected', String(state.side === 'start'));
    $('tabGoal').setAttribute('aria-selected', String(state.side === 'goal'));
    $('tabStartLabel').textContent = replay ? 'Replay' : 'Start';
    $('startCap').textContent = replay ? 'Replay' : 'Start';
    $('editDock').hidden = !!replay;
    $('solSwitch').hidden = !replay;
    $('info').hidden = !!replay;
    $('replayDock').hidden = !replay;
    const hasFat = !!state.start.fat;
    $('fatBtn').setAttribute('aria-pressed', String(hasFat));
    $('fatBtn').title = hasFat ? 'Remove the 2x2 piece' : 'Add the 2x2 piece';
    $('moveTool').hidden = !hasFat;
    $('moveTool').setAttribute('aria-pressed', String(state.moveTool && hasFat));
    const other = otherSide(state.side);
    $('copyBtn').title = `Copy the ${other} board onto the ${state.side} board`;
    $('copyBtn').setAttribute('aria-label', $('copyBtn').title);
    $('fillBtn').title = `Fill the ${state.side} board with ${COLORS[state.color].toLowerCase()}`;
    $('undoBtn').disabled = state.undo.length === 0;
    renderPalette();
    if (replay) showStep(result.step);
  }

  function selectColour(k) {
    state.color = k;
    state.moveTool = false;
    render();
  }

  function setSide(side) {
    state.side = side;
    render();
  }

  // ---------------------------------------------------------------- editing
  function attachEditor(el, which) {
    let painting = false;
    const cellAt = (e) => {
      const t = document.elementFromPoint(e.clientX, e.clientY);
      const cell = t && t.closest('.cell');
      return cell && el.contains(cell) ? cell : null;
    };
    const editable = () => !(which === 'start' && state.mode === 'replay');
    const apply = (cell) => {
      const r = +cell.dataset.r, c = +cell.dataset.c;
      const b = state[which];
      const before = JSON.stringify(b);
      if (state.moveTool && b.fat) placePiece(b, r, c);
      else paint(b, r, c, state.color);
      if (JSON.stringify(b) !== before) renderBoard(el, b);
    };
    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !editable()) return;
      const cell = cellAt(e);
      if (!cell) return;
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      pushUndo();
      if (state.side !== which) {
        state.side = which;
        for (const wrap of document.querySelectorAll('.board-wrap')) wrap.classList.toggle('active', wrap.dataset.side === which);
      }
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
      const b = which === 'start' && state.mode === 'replay' ? result.replay[result.step] : state[which];
      selectColour(b.cells[+cell.dataset.r * 6 + +cell.dataset.c]);
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
    render();
    renderInfo();
    writeHash();
    markStale();
    if (!run) showError('');
  }

  function toggleFat() {
    pushUndo();
    const on = !state.start.fat;
    for (const b of [state.start, state.goal]) {
      if (on) placePiece(b, 2, 2);
      else b.fat = null;
    }
    state.moveTool = on;
    changed();
  }

  function undo() {
    const s = state.undo.pop();
    if (!s) return;
    state.start = s.start;
    state.goal = s.goal;
    state.preset = s.preset;
    leaveReplay();
    changed({ keepPreset: true });
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
      const len = p.moves ? `${p.moves} moves` : `${p.atLeast}+ moves`;
      group.appendChild(new Option(`Puzzle ${p.name} · ${len}${p.fat ? ' · 2x2' : ''}`, p.name));
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
    leaveReplay();
    changed({ keepPreset: true });
  }

  function renderInfo() {
    const el = $('info');
    const p = state.preset && presetByName(state.preset);
    if (p) {   // the game's backdrops, one per world
      const n = (+p.name.split('-')[0] - 1) % 9 + 1;
      document.body.style.setProperty('--backdrop', `url("assets/bkg${n}.webp")`);
    }
    el.classList.remove('warn');
    if (!p) {
      el.textContent = state.start.fat ? 'Custom puzzle with the 2x2 piece' : 'Custom puzzle';
    } else if (p.moves && p.moves <= MAX_DEPTH) {
      el.innerHTML = `Puzzle <b>${p.name}</b> · shortest is ${p.moves} moves · ${p.solutions.toLocaleString()} solutions`;
    } else if (p.moves) {
      el.innerHTML = `Puzzle <b>${p.name}</b> needs ${p.moves} moves, beyond this page's ${MAX_DEPTH}`;
      el.classList.add('warn');
    } else {
      el.innerHTML = `Puzzle <b>${p.name}</b> is unsolved: it needs at least ${p.atLeast} moves`;
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
    if (state.preset) h = `p=${state.preset}&${h}`;
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
    const p = presetByName(q.get('p'));
    if (p) {
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

  const fmtTime = (s) => (s < 10 ? `${s.toFixed(1)} s` : s < 90 ? `${Math.round(s)} s`
    : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`);

  function setProgress(label, fraction) {
    $('solveLabel').textContent = label;
    const fill = $('solveFill');
    fill.classList.toggle('indet', fraction == null);
    fill.style.width = fraction == null ? '' : `${Math.round(fraction * 1000) / 10}%`;
  }

  function showError(text) {
    const el = $('status');
    if (text) {
      el.textContent = text;
      el.classList.add('error');
    } else if (el.classList.contains('error')) {
      el.textContent = '';
      el.classList.remove('error');
    }
  }

  function problemSpec() {
    const a = colourCounts(state.start), b = colourCounts(state.goal);
    if (a.some((v, k) => v !== b[k])) {
      return { error: 'The start and goal boards need the same balls; the red badges show the difference.' };
    }
    const fat = state.start.fat
      ? { start: [state.start.fat.x, state.start.fat.y], goal: [state.goal.fat.x, state.goal.fat.y] } : null;
    return { spec: { start: state.start.cells.slice(), goal: state.goal.cells.slice(), fat } };
  }

  async function solve() {
    if (run) return;
    const { spec, error } = problemSpec();
    if (error) {
      showError(error);
      return;
    }
    showError('');
    const maxDepth = +$('maxDepth').value;
    const threads = +$('threads').value || hardwareThreads;
    run = { workers: [], cancelled: false, t0: performance.now(), boards: { start: cloneBoard(state.start), goal: cloneBoard(state.goal) } };
    const thisRun = run;
    $('solveBtn').classList.add('running');
    $('solveBtn').disabled = true;
    $('cancelBtn').hidden = false;
    let detail = '';
    const tick = setInterval(() => {
      $('status').textContent = `${fmtTime((performance.now() - thisRun.t0) / 1000)}${detail}`;
    }, 200);
    try {
      await workerUrlReady;
      const w0 = warm || new SolverWorker();
      warm = null;
      run.workers.push(w0);
      setProgress('Starting...', null);
      await w0.call('setup', spec);
      setProgress(`Up to ${Math.min(6, maxDepth)} moves...`, null);
      const small = await w0.call('small', { maxDepth });
      if (small.depth >= 0) {
        finish(thisRun, { depth: small.depth, count: small.count, text: small.text, threads: 1, maxDepth });
        return;
      }
      if (maxDepth <= 6) {
        finish(thisRun, { depth: -1, threads: 1, maxDepth });
        return;
      }
      setProgress('Building tables...', null);
      const g = await w0.call('goal4');
      const perWorker = g.memory * 1.5 + 32e6;
      const n = Math.max(1, Math.min(threads, Math.floor(memoryBudget() / perWorker)));
      detail = ` · ${n} thread${n > 1 ? 's' : ''}`;
      for (let i = 1; i < n; i++) run.workers.push(new SolverWorker());
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
          detail = ` · ${n} thread${n > 1 ? 's' : ''}${f > 0.03 && el > 1.5 ? ` · about ${fmtTime(el * (1 - f) / f)} left` : ''}`;
          setProgress(`${d} moves · ${Math.floor(f * 100)}%`, f);
        };
        setProgress(`${d} moves...`, 0);
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
        showError(location.protocol === 'file:'
          ? 'Browsers block workers on file:// pages. Serve the folder instead (python -m http.server in web/).'
          : `The solver failed: ${err.message}`);
      }
    } finally {
      clearInterval(tick);
      for (const w of thisRun.workers) w.terminate();
      if (run === thisRun) run = null;
      $('solveBtn').classList.remove('running');
      $('solveBtn').disabled = false;
      $('cancelBtn').hidden = true;
      setProgress('Solve', 0);
      if (!$('status').classList.contains('error')) $('status').textContent = thisRun.summary || '';
      warmUp();
    }
  }

  function cancel() {
    if (!run) return;
    run.cancelled = true;
    run.summary = 'Cancelled';
    for (const w of run.workers) w.terminate();
  }

  // ---------------------------------------------------------------- results
  function finish(r, res) {
    if (r.cancelled) return;
    const secs = (performance.now() - r.t0) / 1000;
    const lines = res.text ? res.text.split('\n').filter(Boolean).sort() : [];
    result = { ...res, lines, secs, boards: r.boards, shown: 0, selected: -1, step: 0, playing: null };
    const threads = `${res.threads} thread${res.threads > 1 ? 's' : ''}`;
    r.summary = `Solved in ${fmtTime(secs)} on ${threads}`;
    $('results').hidden = false;
    $('intro').hidden = true;
    if (res.depth < 0) {
      $('resultTitle').textContent = `No solution in ${res.maxDepth} moves`;
      $('resultSub').textContent = `Every sequence of up to ${res.maxDepth} moves was checked.`;
    } else if (res.depth === 0) {
      $('resultTitle').textContent = 'Already solved';
      $('resultSub').textContent = 'The start board is the goal board.';
    } else {
      const n = res.count;
      $('resultTitle').textContent = `${res.depth} move${res.depth > 1 ? 's' : ''}`;
      $('resultSub').textContent = `${n.toLocaleString()} shortest solution${n > 1 ? 's' : ''} · none shorter`
        + (lines.length < n ? ` · ${lines.length.toLocaleString()} listed` : '') + ` · ${fmtTime(secs)}`;
    }
    $('copyAllBtn').hidden = $('downloadBtn').hidden = lines.length === 0;
    $('solutionList').replaceChildren();
    showMore();
    markStale();
    if (lines.length) {
      selectSolution(0);
      if (innerWidth < 1024) document.querySelector('.stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      leaveReplay();
      render();
    }
  }

  function markStale() {
    if (!result) return;
    const same = sameBoard(result.boards.start, state.start) && sameBoard(result.boards.goal, state.goal);
    $('results').classList.toggle('is-stale', !same);
    $('staleNote').hidden = same;
  }

  function showMore() {
    const list = $('solutionList');
    const end = Math.min(result.lines.length, result.shown + LIST_PAGE);
    const frag = document.createDocumentFragment();
    for (let i = result.shown; i < end; i++) {
      const li = document.createElement('li');
      li.dataset.i = i;
      li.setAttribute('aria-selected', 'false');
      li.innerHTML = `<span class="k">${i + 1}</span><span></span>`;
      li.lastChild.textContent = result.lines[i];
      frag.appendChild(li);
    }
    list.appendChild(frag);
    result.shown = end;
    const left = result.lines.length - end;
    $('moreBtn').hidden = left <= 0;
    $('moreBtn').textContent = `Show ${Math.min(left, LIST_PAGE).toLocaleString()} more (${left.toLocaleString()} left)`;
  }

  function selectSolution(i) {
    if (!result || !result.lines.length) return;
    i = (i + result.lines.length) % result.lines.length;
    while (i >= result.shown) showMore();
    stopPlay();
    const prev = $('solutionList').querySelector('[aria-selected="true"]');
    if (prev) prev.setAttribute('aria-selected', 'false');
    const li = $('solutionList').querySelector(`[data-i="${i}"]`);
    if (li) {
      li.setAttribute('aria-selected', 'true');
      // keep it visible when the list scrolls by itself (side panel), without moving the page
      const list = $('solutionList');
      if (list.scrollHeight > list.clientHeight + 1) {
        const top = li.offsetTop, bottom = top + li.offsetHeight;
        if (top < list.scrollTop) list.scrollTop = top - 4;
        else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight + 4;
      }
    }
    const moves = result.lines[i].split(' ').map(parseMove);
    const boards = [cloneBoard(result.boards.start)];
    for (const mv of moves) boards.push(applyMove(boards[boards.length - 1], mv));
    result.selected = i;
    result.moves = moves;
    result.replay = boards;
    result.step = 0;
    const chips = $('replayChips');
    chips.replaceChildren();
    moves.forEach((mv, k) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = mv.name;
      b.title = describeMove(mv);
      b.addEventListener('click', () => showStep(k));
      chips.appendChild(b);
    });
    $('solLabel').textContent = `Solution ${(i + 1).toLocaleString()} of ${result.lines.length.toLocaleString()}`;
    $('prevSol').disabled = $('nextSol').disabled = result.lines.length < 2;
    state.mode = 'replay';
    state.side = 'start';
    state.moveTool = false;
    render();
  }

  function leaveReplay() {
    stopPlay();
    state.mode = 'edit';
  }

  function showStep(k, animate = false) {
    if (!result || !result.replay) return;
    const n = result.moves.length;
    k = Math.max(0, Math.min(n, k));
    const from = result.step;
    result.step = k;
    let anim = null;
    if (animate && k === from + 1) anim = { mv: result.moves[from], dir: 1 };
    if (animate && k === from - 1) anim = { mv: result.moves[k], dir: -1 };
    const next = k < n ? result.moves[k] : null;
    renderBoard($('startBoard'), result.replay[k], { band: next, anim });
    renderThumb($('thumbStart'), result.replay[k]);
    $('stepText').innerHTML = next
      ? `Move ${k + 1} of ${n}: <b>${describeMove(next)}</b>`
      : `<b>Solved</b> in ${n} move${n > 1 ? 's' : ''}`;
    [...$('replayChips').children].forEach((c, i) => {
      c.classList.toggle('done', i < k);
      c.classList.toggle('now', i === k);
    });
    const now = $('replayChips').children[Math.min(k, n - 1)];
    const strip = $('replayChips');
    if (now && strip.scrollWidth > strip.clientWidth) {
      strip.scrollTo({ left: now.offsetLeft - strip.clientWidth / 2 + now.offsetWidth / 2, behavior: 'smooth' });
    }
    $('firstBtn').disabled = $('prevBtn').disabled = k === 0;
    $('nextBtn').disabled = $('lastBtn').disabled = k === n;
  }

  function stopPlay() {
    if (result && result.playing) {
      clearInterval(result.playing);
      result.playing = null;
    }
    setIcon($('playBtn'), 'play');
    $('playBtn').setAttribute('aria-label', 'Play');
  }

  function togglePlay() {
    if (!result || !result.replay) return;
    if (result.playing) {
      stopPlay();
      return;
    }
    if (result.step >= result.moves.length) showStep(0);
    setIcon($('playBtn'), 'pause');
    $('playBtn').setAttribute('aria-label', 'Pause');
    result.playing = setInterval(() => {
      if (result.step >= result.moves.length) {
        stopPlay();
        return;
      }
      showStep(result.step + 1, true);
    }, 900);
  }

  function download(name, text) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function flash(btn, iconName = 'check') {
    setIcon(btn, iconName);
    setTimeout(() => setIcon(btn, btn.dataset.icon), 1300);
  }

  // ---------------------------------------------------------------- wiring
  function init() {
    document.querySelectorAll('[data-icon]').forEach((el) => setIcon(el, el.dataset.icon));
    fillPresets();
    for (let d = 1; d <= MAX_DEPTH; d++) $('maxDepth').appendChild(new Option(`≤ ${d} move${d > 1 ? 's' : ''}`, d));
    $('maxDepth').value = MAX_DEPTH;
    $('threads').appendChild(new Option(`Auto (${hardwareThreads})`, 0));
    for (const t of [1, 2, 4, 8, 12, 16, 24, 32]) {
      if (t < hardwareThreads) $('threads').appendChild(new Option(String(t), t));
    }

    attachEditor($('startBoard'), 'start');
    attachEditor($('goalBoard'), 'goal');

    $('preset').addEventListener('change', (e) => {
      if (e.target.value) loadPreset(e.target.value);
      else { state.preset = ''; changed(); }
      e.target.blur();
    });
    $('tabStart').addEventListener('click', () => setSide('start'));
    $('tabGoal').addEventListener('click', () => setSide('goal'));
    $('fatBtn').addEventListener('click', toggleFat);
    $('moveTool').addEventListener('click', () => {
      state.moveTool = !state.moveTool;
      render();
    });
    $('undoBtn').addEventListener('click', undo);
    $('fillBtn').addEventListener('click', () => {
      pushUndo();
      state[state.side].cells.fill(state.color);
      changed();
    });
    $('copyBtn').addEventListener('click', () => {
      pushUndo();
      state[state.side] = cloneBoard(state[otherSide(state.side)]);
      changed();
    });
    $('swapBtn').addEventListener('click', () => {
      pushUndo();
      [state.start, state.goal] = [state.goal, state.start];
      changed();
    });
    $('shareBtn').addEventListener('click', async () => {
      writeHash();
      try {
        await navigator.clipboard.writeText(location.href);
        flash($('shareBtn'));
      } catch {
        prompt('Copy this link:', location.href);
      }
    });
    $('helpBtn').addEventListener('click', () => $('helpDialog').showModal());
    $('helpDialog').addEventListener('click', (e) => { if (e.target === $('helpDialog')) $('helpDialog').close(); });

    $('solveBtn').addEventListener('click', solve);
    $('cancelBtn').addEventListener('click', cancel);
    $('moreBtn').addEventListener('click', showMore);
    $('solutionList').addEventListener('click', (e) => {
      const li = e.target.closest('li');
      if (!li) return;
      selectSolution(+li.dataset.i);
      if (innerWidth < 1024) document.querySelector('.stage').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    $('editBtn').addEventListener('click', () => { leaveReplay(); render(); });
    $('prevSol').addEventListener('click', () => selectSolution(result.selected - 1));
    $('nextSol').addEventListener('click', () => selectSolution(result.selected + 1));
    $('firstBtn').addEventListener('click', () => { stopPlay(); showStep(0); });
    $('prevBtn').addEventListener('click', () => { stopPlay(); showStep(result.step - 1, true); });
    $('nextBtn').addEventListener('click', () => { stopPlay(); showStep(result.step + 1, true); });
    $('lastBtn').addEventListener('click', () => { stopPlay(); showStep(Infinity); });
    $('playBtn').addEventListener('click', togglePlay);
    $('copyAllBtn').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(result.lines.join('\n') + '\n');
        flash($('copyAllBtn'));
      } catch { /* clipboard blocked */ }
    });
    $('downloadBtn').addEventListener('click', () => {
      download(`${state.preset || 'puzzle'}_${result.depth}moves_${result.count}.txt`, result.lines.join('\n') + '\n');
    });

    document.addEventListener('keydown', (e) => {
      if (e.target.closest('select, input, textarea, dialog')) return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undo();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (state.mode === 'replay' && result && result.replay) {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          stopPlay();
          showStep(result.step + (e.key === 'ArrowRight' ? 1 : -1), true);
        } else if (e.key === ' ') {
          e.preventDefault();
          togglePlay();
        } else if (e.key === 'Escape') {
          leaveReplay();
          render();
        }
        return;
      }
      if (/^[1-8]$/.test(e.key)) selectColour(+e.key - 1);
      else if (e.key.toLowerCase() === 'm' && state.start.fat) {
        state.moveTool = !state.moveTool;
        render();
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

  init();
})();
