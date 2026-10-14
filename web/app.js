// web/app.js - board editor, the solver workers, results and the guided replay.
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
    unpiece: '<rect x="3" y="3" width="13" height="13" rx="3.5"/><path d="m15 15 6 6m0-6-6 6"/>',
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
    placing: false,         // the palette's 2x2 tool: taps place the big chuzzle
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
  const pieceColour = (b) => (b.fat ? b.cells[b.fat.y * 6 + b.fat.x] : null);

  function paint(board, r, c, color) {
    if (inPiece(board.fat, r, c)) {
      for (const i of pieceCells(board.fat)) board.cells[i] = color;
    } else {
      board.cells[r * 6 + c] = color;
    }
  }

  // The 2x2 chuzzle with its top-left at (r, c), kept inside the board. Its cells take
  // its colour; the cells it leaves keep theirs.
  function placePieceAt(board, r, c, color) {
    board.fat = { x: Math.min(c, 4), y: Math.min(r, 4) };
    for (const i of pieceCells(board.fat)) board.cells[i] = color;
  }

  // The 2x2 tool: moves this board's piece there, or places a new one (in the selected
  // colour). A puzzle has the piece on both boards, so the other board gets one too.
  function placeFat(which, r, c) {
    const b = state[which], o = state[otherSide(which)];
    const color = pieceColour(b) ?? pieceColour(o) ?? state.color;
    placePieceAt(b, r, c, color);
    if (!o.fat) placePieceAt(o, r, c, color);
  }

  // ---------------------------------------------------------------- moves
  // "R14" row 1 right 4, "C32" column 3 down 2, "R453" / "C231" the 2x2 chuzzle's two lanes.
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

  const laneName = (mv) => (mv.lanes.length === 2
    ? `${mv.col ? 'columns' : 'rows'} ${mv.lanes[0] + 1}+${mv.lanes[1] + 1}`
    : `${mv.col ? 'column' : 'row'} ${mv.lanes[0] + 1}`);
  const describeMove = (mv) => `${laneName(mv)} ${mv.col ? 'down' : 'right'} ${mv.n}`;

  // How to make a move by hand: drag the short way round (right 5 = left 1), d cells along
  // the lane from position s, across the lane at `cross`. A 2x2 lane is grabbed by the big
  // chuzzle itself when the drag fits.
  function dragOf(mv, board) {
    const d = mv.n <= 3 ? mv.n : mv.n - 6;
    const lo = d > 0 ? 0 : -d, hi = d > 0 ? 5 - d : 5;
    let s = Math.max(lo, Math.min(hi, Math.round((5 - d) / 2)));
    let cross = mv.lanes[0];
    if (mv.lanes.length === 2) {
      cross += 0.5;
      const f = board.fat;
      const along = f ? (mv.col ? f.y : f.x) + 0.5 : -1;
      if (f && along >= lo && along <= hi) s = along;
    }
    return { d, s, cross };
  }

  function dragText(mv, g) {
    const dir = mv.col ? (g.d > 0 ? 'down' : 'up') : (g.d > 0 ? 'right' : 'left');
    return `drag ${laneName(mv)} ${dir} ${Math.abs(g.d)}`;
  }

  // ---------------------------------------------------------------- board rendering
  // opts.band: highlight the lane(s) a move will shift.
  function renderBoard(el, board, opts = {}) {
    const frag = document.createDocumentFragment();
    const f = board.fat;
    if (opts.band) {
      const mv = opts.band;
      const band = document.createElement('div');
      band.className = 'band-hl';
      const span = `${Math.min(...mv.lanes) + 1} / span ${mv.lanes.length}`;
      band.style.gridRow = mv.col ? '1 / -1' : span;
      band.style.gridColumn = mv.col ? span : '1 / -1';
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
        d.setAttribute('aria-label', `${piece ? '2x2 chuzzle, ' : ''}row ${r + 1}, column ${c + 1}: ${COLORS[color]}`);
        frag.appendChild(d);
      }
    }
    el.replaceChildren(frag);
  }

  // A chuzzle's sprite: colour row, and an eye shape, pupil colour and sway phase that
  // stay with it (like the game's random ones, but fixed per chuzzle).
  function setChuzzle(el, color, id) {
    el.style.setProperty('--k', color);
    el.style.setProperty('--es', (id * 7 + 3) % 4);
    el.style.setProperty('--ep', (id * 5 + 1) % 6);
    el.style.setProperty('--ph', ((id * 37) % 100) / 100);
  }

  // Sets up a lane slide on a board already showing the state after the move: every moved
  // ball starts d cells back and set(p) carries it to its place (in the game's stretched
  // frame for the direction); a ball that wrapped round the edge slides in from outside
  // while a copy of it slides out.
  function prepareSlide(el, mv, d) {
    const any = el.querySelector('.cell:not(.piece)') || el.querySelector('.cell');
    const gap = parseFloat(getComputedStyle(el).columnGap) || 0;
    const pitch = (any.classList.contains('piece') ? (any.offsetWidth - gap) / 2 : any.offsetWidth) + gap;
    const seg = mv.col ? (d > 0 ? 0 : 3) : (d > 0 ? 2 : 1);      // down, left, right, up
    const moving = [];
    for (const cell of [...el.querySelectorAll('.cell')]) {
      const r = +cell.dataset.r, c = +cell.dataset.c;
      const piece = cell.classList.contains('piece');
      const lanes = mv.col ? (piece ? [c, c + 1] : [c]) : (piece ? [r, r + 1] : [r]);
      if (!lanes.some((l) => mv.lanes.includes(l))) continue;
      const j = mv.col ? r : c;
      const w = piece ? 2 : 1;
      moving.push([cell, -d, 0]);
      if (d > 0 ? j - d < 0 : j - d + w - 1 > 5) {
        const ghost = cell.cloneNode(true);
        ghost.classList.add('ghost');
        ghost.removeAttribute('role');
        el.appendChild(ghost);
        moving.push([ghost, d > 0 ? 6 - d : -6 - d, d > 0 ? 6 : -6]);
      }
    }
    for (const [node] of moving) {
      node.style.setProperty('--f', (node.classList.contains('piece') ? 10 : 20) + seg);
      node.classList.add('mv');
    }
    const tr = (k) => (mv.col ? `translateY(${k * pitch}px)` : `translateX(${k * pitch}px)`);
    const set = (p) => { for (const [node, a, b] of moving) node.style.transform = tr(a + (b - a) * p); };
    set(0);
    return {
      set,
      finish() {
        el.querySelectorAll('.ghost').forEach((g) => g.remove());
        for (const [node] of moving) {
          node.style.transform = '';
          node.classList.remove('mv');
        }
      },
    };
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
      const big = document.createElement('button');
      big.type = 'button';
      big.className = 'swatch big';
      big.id = 'fatSwatch';
      big.setAttribute('role', 'radio');
      big.innerHTML = '<span class="n">2x2</span>';
      big.addEventListener('click', selectFatTool);
      el.appendChild(big);
    }
    COLORS.forEach((name, k) => {
      const s = el.children[k];
      const ok = a[k] === b[k];
      s.setAttribute('aria-checked', String(!state.placing && state.color === k));
      s.setAttribute('aria-label', `${name}: ${a[k]} on start, ${b[k]} on goal`);
      s.title = `${name} (key ${k + 1})${ok ? '' : ` - ${a[k]} on start, ${b[k]} on goal`}`;
      const n = s.firstChild;
      n.className = ok ? 'n' : 'n bad';
      n.textContent = a[k] || b[k] ? (ok ? a[k] : `${a[k]}≠${b[k]}`) : '';
    });
    const big = $('fatSwatch');
    setChuzzle(big, pieceColour(state.start) ?? state.color, 40);
    big.setAttribute('aria-checked', String(state.placing));
    big.setAttribute('aria-label', '2x2 Chuzzle: tap a cell to place it');
    big.title = state.start.fat ? '2x2 Chuzzle: tap a cell to move it (key 9)' : '2x2 Chuzzle: tap a cell to place it (key 9)';
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
      b.classList.toggle('placing', !replay && state.placing);
      b.classList.toggle('readonly', side === 'start' && !!replay);
    }
    renderThumb($('thumbStart'), replay ? result.replay[result.step] : state.start);
    renderThumb($('thumbGoal'), state.goal);
    $('tabStart').setAttribute('aria-selected', String(state.side === 'start'));
    $('tabGoal').setAttribute('aria-selected', String(state.side === 'goal'));
    $('tabStartLabel').textContent = replay ? 'Replay' : 'Start';
    $('startCap').textContent = replay ? 'Replay' : 'Start';
    $('editDock').hidden = !!replay;
    $('replayDock').hidden = !replay;
    $('solSwitch').hidden = !replay;
    $('info').hidden = !!replay;
    $('guide').hidden = !replay;
    $('unfatBtn').hidden = !state.start.fat;
    const other = otherSide(state.side);
    $('copyBtn').title = `Copy the ${other} board onto the ${state.side} board`;
    $('copyText').textContent = `Copy ${other}`;
    $('fillBtn').title = `Fill the ${state.side} board with ${COLORS[state.color].toLowerCase()}`;
    $('undoBtn').disabled = state.undo.length === 0;
    renderPalette();
    if (replay) showStep(result.step);
  }

  function selectColour(k) {
    state.color = k;
    state.placing = false;
    render();
  }

  function selectFatTool() {
    state.placing = !state.placing;
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
      if (state.placing) placeFat(which, r, c);
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

  function removeFat() {
    pushUndo();
    state.start.fat = null;
    state.goal.fat = null;
    state.placing = false;
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
    state.placing = false;
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
      el.textContent = state.start.fat ? 'Custom puzzle with the 2x2 Chuzzle' : 'Custom puzzle';
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
    result = { ...res, lines, secs, boards: r.boards, shown: 0, selected: -1, step: 0, playing: false };
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
    interrupt();
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
    guide.at = null;
    const chips = $('replayChips');
    chips.replaceChildren();
    moves.forEach((mv, k) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = mv.name;
      b.title = describeMove(mv);
      b.addEventListener('click', () => { interrupt(); showStep(k); });
      chips.appendChild(b);
    });
    $('solLabel').textContent = `Solution ${(i + 1).toLocaleString()} of ${result.lines.length.toLocaleString()}`;
    $('prevSol').disabled = $('nextSol').disabled = result.lines.length < 2;
    state.mode = 'replay';
    state.side = 'start';
    state.placing = false;
    render();
  }

  function leaveReplay() {
    interrupt();
    state.mode = 'edit';
  }

  // ---------------------------------------------------------------- the guided replay
  // The game's wand cursor shows each move: it glides to a chuzzle in the lane, presses,
  // drags the lane (which slides with it) the short way round, and lets go. Between moves
  // an arrow on the board shows the next drag.
  const HOT = { x: 12, y: 11 };               // the star's tip in the 52 px wand sprite
  const guide = { at: null };                 // where the wand is, in the start figure's pixels
  let animToken = 0;
  const interrupt = () => {
    animToken++;
    if (result) result.playing = false;
    setIcon($('playBtn'), 'play');
    $('playBtn').setAttribute('aria-label', 'Play');
    return animToken;
  };
  const wait = (ms, token) => new Promise((res) => setTimeout(() => res(token === animToken), reducedMotion.matches ? 0 : ms));
  function tween(ms, fn, token) {
    return new Promise((resolve) => {
      if (reducedMotion.matches) {
        fn(1);
        resolve(token === animToken);
        return;
      }
      const t0 = performance.now();
      const frame = (now) => {
        if (token !== animToken) {
          resolve(false);
          return;
        }
        const p = Math.min(1, (now - t0) / ms);
        fn(p);
        if (p < 1) requestAnimationFrame(frame);
        else resolve(true);
      };
      requestAnimationFrame(frame);
    });
  }
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
  const easeOut = (t) => 1 - (1 - t) ** 3;
  const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

  // A spot on the start board (lane position `along`, across the lane at `cross`), in the
  // pixels of the board's figure, where the guide layer sits.
  function boardPoint(mv, along, cross) {
    const b = $('startBoard');
    const pad = parseFloat(getComputedStyle(b).paddingLeft) || 0;
    const any = b.querySelector('.cell:not(.piece)');
    const cell = any ? any.offsetWidth : 40;
    const pitch = cell + (parseFloat(getComputedStyle(b).columnGap) || 0);
    const ox = b.offsetLeft + b.clientLeft + pad + cell / 2;
    const oy = b.offsetTop + b.clientTop + pad + cell / 2;
    const r = mv.col ? along : cross, c = mv.col ? cross : along;
    return { x: ox + c * pitch, y: oy + r * pitch };
  }

  function placeCursor(p, down = false) {
    guide.at = p;
    const el = $('guideCursor');
    el.style.transform = `translate(${p.x - HOT.x}px, ${p.y - HOT.y}px)`;
    el.classList.toggle('down', down);
  }

  function pulseAt(p) {
    const ring = $('guideRing');
    ring.style.left = `${p.x}px`;
    ring.style.top = `${p.y}px`;
    ring.classList.remove('pulse');
    void ring.offsetWidth;
    ring.classList.add('pulse');
  }

  function drawArrow(p0, p1) {
    const path = $('guidePath');
    const len = Math.hypot(p1.x - p0.x, p1.y - p0.y) || 1;
    const back = Math.min(10, len / 3);       // leave room for the arrowhead
    const e = { x: p1.x - (p1.x - p0.x) / len * back, y: p1.y - (p1.y - p0.y) / len * back };
    path.setAttribute('d', `M${p0.x} ${p0.y}L${e.x} ${e.y}`);
    $('guideArrow').classList.remove('off');
  }

  // The still view of step k: the board, the next lane lit, the arrow and the wand on it.
  function showStep(k, keepCursor = false) {
    if (!result || !result.replay) return;
    const n = result.moves.length;
    k = Math.max(0, Math.min(n, k));
    result.step = k;
    const next = k < n ? result.moves[k] : null;
    renderBoard($('startBoard'), result.replay[k], { band: next });
    renderThumb($('thumbStart'), result.replay[k]);
    if (next) {
      const g = dragOf(next, result.replay[k]);
      const p0 = boardPoint(next, g.s, g.cross), p1 = boardPoint(next, g.s + g.d, g.cross);
      drawArrow(p0, p1);
      if (!keepCursor || !guide.at) placeCursor(p0);
      $('guideCursor').hidden = false;
      $('stepText').innerHTML = `Move ${k + 1} of ${n}: <b>${dragText(next, g)}</b>${next.n > 3 ? ` <span class="nm">(${next.name})</span>` : ''}`;
    } else {
      $('guideArrow').classList.add('off');
      $('guideCursor').hidden = !keepCursor;
      $('stepText').innerHTML = `<b>Solved</b> in ${n} move${n > 1 ? 's' : ''}`;
    }
    [...$('replayChips').children].forEach((c, i) => {
      c.classList.toggle('done', i < k);
      c.classList.toggle('now', i === k);
    });
    const strip = $('replayChips');
    const now = strip.children[Math.min(k, n - 1)];
    if (now && strip.scrollWidth > strip.clientWidth) {
      strip.scrollTo({ left: now.offsetLeft - strip.clientWidth / 2 + now.offsetWidth / 2, behavior: 'smooth' });
    }
    $('firstBtn').disabled = $('prevBtn').disabled = k === 0;
    $('nextBtn').disabled = $('lastBtn').disabled = k === n;
  }

  // Plays move k with the wand: glide to the lane, press, drag (the lane follows), let go.
  async function animateMove(k, token) {
    const mv = result.moves[k];
    const g = dragOf(mv, result.replay[k]);
    const p0 = boardPoint(mv, g.s, g.cross), p1 = boardPoint(mv, g.s + g.d, g.cross);
    const from = guide.at || { x: p0.x + 30, y: p0.y + 70 };
    $('guideCursor').hidden = false;
    const dist = Math.hypot(p0.x - from.x, p0.y - from.y);
    if (dist > 2 && !(await tween(Math.min(700, 280 + dist * 1.6), (p) => placeCursor(lerp(from, p0, easeOut(p))), token))) return false;
    $('guideArrow').classList.add('off');
    placeCursor(p0, true);
    pulseAt(p0);
    if (!(await wait(220, token))) return false;
    renderBoard($('startBoard'), result.replay[k + 1], { band: mv });
    const slide = prepareSlide($('startBoard'), mv, g.d);
    const ok = await tween(260 * Math.abs(g.d) + 240, (p) => {
      const e = easeInOut(p);
      slide.set(e);
      placeCursor(lerp(p0, p1, e), true);
    }, token);
    slide.finish();
    if (!ok) return false;
    placeCursor(p1, true);
    if (!(await wait(160, token))) return false;
    placeCursor(p1, false);
    showStep(k + 1, true);
    return true;
  }

  // Undoes move k-1 with a quick slide back (no wand).
  async function animateBack(k, token) {
    const mv = result.moves[k - 1];
    const g = dragOf(mv, result.replay[k - 1]);
    $('guideArrow').classList.add('off');
    renderBoard($('startBoard'), result.replay[k - 1], { band: mv });
    const slide = prepareSlide($('startBoard'), mv, -g.d);
    const ok = await tween(140 * Math.abs(g.d) + 160, (p) => slide.set(easeInOut(p)), token);
    slide.finish();
    if (ok) showStep(k - 1);
    return ok;
  }

  async function next() {
    if (!result || result.step >= result.moves.length) return;
    const token = interrupt();
    if (!(await animateMove(result.step, token)) && token === animToken) showStep(result.step);
  }

  async function prev() {
    if (!result || result.step <= 0) return;
    const token = interrupt();
    if (!(await animateBack(result.step, token)) && token === animToken) showStep(result.step);
  }

  async function togglePlay() {
    if (!result || !result.replay) return;
    if (result.playing) {
      interrupt();
      showStep(result.step);
      return;
    }
    const token = interrupt();
    if (result.step >= result.moves.length) showStep(0);
    result.playing = true;
    setIcon($('playBtn'), 'pause');
    $('playBtn').setAttribute('aria-label', 'Pause');
    while (token === animToken && result.step < result.moves.length) {
      if (!(await animateMove(result.step, token))) break;
      if (!(await wait(600, token))) break;
    }
    if (token === animToken) interrupt();
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
    $('unfatBtn').addEventListener('click', removeFat);
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
    $('firstBtn').addEventListener('click', () => { interrupt(); showStep(0); });
    $('prevBtn').addEventListener('click', prev);
    $('nextBtn').addEventListener('click', next);
    $('lastBtn').addEventListener('click', () => { interrupt(); showStep(Infinity); });
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
    addEventListener('hashchange', () => {      // a pasted link to another puzzle
      const before = location.hash;
      pushUndo();
      if (!readHash()) return;
      leaveReplay();
      changed({ keepPreset: true });
      if (location.hash !== before) history.replaceState(null, '', before);
    });
    let resizeTimer = 0;
    addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.mode === 'replay' && result && result.replay && !result.playing) showStep(result.step);
      }, 120);
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
        if (e.key === 'ArrowRight') {
          e.preventDefault();
          next();
        } else if (e.key === 'ArrowLeft') {
          e.preventDefault();
          prev();
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
      else if (e.key === '9' || e.key.toLowerCase() === 'm') selectFatTool();
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
