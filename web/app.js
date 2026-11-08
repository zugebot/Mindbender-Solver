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
    list: '<path d="M8 6h13M8 12h13M8 18h13"/><path d="M3 6h.01M3 12h.01M3 18h.01"/>',
    up: '<path d="m18 15-6-6-6 6"/>',
    shuffle: '<path d="M3 7h3.5c4.5 0 6.5 10 11 10H21"/><path d="m18 14 3 3-3 3"/><path d="M3 17h3.5c1.6 0 2.8-1.3 3.8-3"/><path d="M13.7 10c1-1.7 2.2-3 3.8-3H21"/><path d="m18 4 3 3-3 3"/>',
    reset: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    grid: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/>',
    minus: '<path d="M6 12h12"/>',
    plus: '<path d="M12 6v12M6 12h12"/>',
  };
  // White symbols over a dark outline, the way the game letters its buttons.
  const icon = (name) => `<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><g class="ol">${ICONS[name]}</g><g class="fg">${ICONS[name]}</g></svg>`;
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
    tool: 'paint',          // the goal's tool: 'paint' (the colour) or 'piece' (place the 2x2)
    undo: [],
    preset: '',
    follows: true,          // the start is still the goal's copy (Reset has nothing to do)
    picked: null,           // the start chuzzle tapped first for a swap: { side, i }
    custom: null,           // the custom puzzle, kept while a game puzzle is open
    coach: 0,               // the tutorial's step (0: not showing)
    side: 'goal',           // the board being edited (the one shown on phones)
    mode: 'edit',           // 'edit' | 'replay'
    maxDepth: MAX_DEPTH,    // the longest solution the search looks for
  };

  // ids: which chuzzle sits in each cell (replay moves them, so each keeps its eyes)
  const cloneBoard = (b) => ({ cells: b.cells.slice(), fat: b.fat ? { ...b.fat } : null, ids: b.ids ? b.ids.slice() : undefined });
  const sameBoard = (a, b) => a.cells.every((v, i) => v === b.cells[i])
    && JSON.stringify(a.fat) === JSON.stringify(b.fat);
  const otherSide = (s) => (s === 'start' ? 'goal' : 'start');

  function pushUndo() {
    state.undo.push({ start: cloneBoard(state.start), goal: cloneBoard(state.goal), preset: state.preset, follows: state.follows });
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

  // The 2x2 tool: moves this board's piece there, or places a new one in the selected colour.
  // A puzzle has the piece on both boards, so the other board gets one too.
  function placeFat(which, r, c) {
    const b = state[which], o = state[otherSide(which)];
    const color = pieceColour(b) ?? pieceColour(o) ?? state.color;
    placePieceAt(b, r, c, color);
    if (!o.fat) placePieceAt(o, r, c, color);
  }

  // The start and goal no longer hold the same chuzzles (a board was painted).
  const startMismatch = () => !state.follows && (!!state.start.fat !== !!state.goal.fat
    || colourCounts(state.start).some((n, k) => n !== colourCounts(state.goal)[k]));

  // The start is the goal's chuzzles rearranged (dragged, swapped or mixed, never painted):
  // start.ids[k] = the goal cell whose chuzzle sits in start cell k. Painting the goal repaints
  // that chuzzle on the start too, so the two always hold the same chuzzles. Placing or moving
  // the goal's 2x2 keeps the start's arrangement (relinkPiece).
  const copyOfGoal = () => {
    const b = cloneBoard(state.goal);
    b.ids = b.cells.map((_, k) => k);
    return b;
  };
  const pieceLinked = (s, g) => !s.fat === !g.fat
    && (!s.fat || pieceCells(s.fat).every((k) => pieceCells(g.fat).includes(s.ids[k])));
  function syncStart() {
    const s = state.start;
    if (!state.follows && !s.ids) {
      linkStart();
      return;
    }
    if (!state.follows && s.fat && state.goal.fat && !pieceLinked(s, state.goal)) relinkPiece(s, state.goal);
    if (state.follows || !pieceLinked(s, state.goal)) {
      state.start = copyOfGoal();
      state.follows = true;
      return;
    }
    s.cells = s.ids.map((i) => state.goal.cells[i]);
  }
  // The goal's 2x2 placed or moved under an arranged start: the start's 2x2 takes the goal's
  // new 2x2 cells, and the start chuzzles that were those goal chuzzles take the ones its 2x2
  // had. Everything else keeps its place.
  function relinkPiece(s, g) {
    const gp = pieceCells(g.fat), sp = pieceCells(s.fat);
    const inPieceWrong = sp.filter((k) => !gp.includes(s.ids[k]));
    const outsideTaken = s.ids.map((id, k) => (!sp.includes(k) && gp.includes(id) ? k : -1)).filter((k) => k >= 0);
    inPieceWrong.forEach((a, n) => {
      const b = outsideTaken[n];
      [s.ids[a], s.ids[b]] = [s.ids[b], s.ids[a]];
    });
  }
  // A start from a link (or saved before the start was only rearranged): which goal chuzzle
  // is which. The 2x2's cells go to the goal's 2x2, the rest to goal chuzzles of their colour
  // in reading order; a start chuzzle with no goal chuzzle of its colour left takes one of the
  // goal's leftovers and its colour. So the start always ends up a rearrangement of the goal.
  function linkStart() {
    const s = state.start, g = state.goal;
    if (!!s.fat !== !!g.fat) {
      state.start = copyOfGoal();
      state.follows = true;
      return;
    }
    const ids = Array(36).fill(-1);
    const gp = g.fat ? pieceCells(g.fat) : [], sp = s.fat ? pieceCells(s.fat) : [];
    sp.forEach((k, m) => { ids[k] = gp[m]; });
    const free = g.cells.map((col, k) => (gp.includes(k) ? -1 : k)).filter((k) => k >= 0);
    for (let k = 0; k < 36; k++) {
      if (sp.includes(k)) continue;
      const n = free.findIndex((j) => g.cells[j] === s.cells[k]);
      if (n >= 0) ids[k] = free.splice(n, 1)[0];
    }
    for (let k = 0; k < 36; k++) if (ids[k] < 0) ids[k] = free.shift();
    s.ids = ids;
    s.cells = ids.map((i) => g.cells[i]);
    state.follows = ids.every((v, k) => v === k);
  }

  // A lane move (Scramble uses them): lane `lane` of rows (col false) or columns shifted by n
  // (right / down); a lane through the 2x2 takes the piece's other lane with it, and the
  // piece never ends straddling the edge (the solver's moveValid). null if not allowed.
  function laneMove(board, col, lane, n) {
    n = ((n % 6) + 6) % 6;
    const f = board.fat;
    let lanes = [lane];
    if (f) {
      const a = col ? f.x : f.y;
      if (lane === a || lane === a + 1) {
        lanes = [a, a + 1];
        const along = col ? f.y : f.x;
        if ((along + n) % 6 === 5) return null;
      }
    }
    return applyMove(board, { col, lanes, n });
  }

  // 30 random moves (no lane twice in a row), the game's Scrambulator in spirit.
  function scrambled(board) {
    let b = cloneBoard(board), last = -1;
    for (let k = 0; k < 30; k++) {
      let col, lane;
      do {
        col = Math.random() < 0.5;
        lane = Math.floor(Math.random() * 6);
      } while ((col ? 6 : 0) + lane === last);
      last = (col ? 6 : 0) + lane;
      const next = laneMove(b, col, lane, 1 + Math.floor(Math.random() * 5));
      if (next) b = next;
      else k--;
    }
    return b;
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
        if (state.picked && el.id === `${state.picked.side}Board` && state.mode === 'edit' && state.picked.i === r * 6 + c) d.classList.add('picked');
        d.dataset.r = r;
        d.dataset.c = c;
        const id = board.ids ? board.ids[r * 6 + c] : r * 6 + c;
        d.dataset.k = color;            // drawn on the board's canvas
        d.dataset.eye = `c:${id}`;
        d.style.gridRow = `${r + 1} / span ${piece ? 2 : 1}`;
        d.style.gridColumn = `${c + 1} / span ${piece ? 2 : 1}`;
        d.setAttribute('role', 'gridcell');
        d.setAttribute('aria-label', `${piece ? '2x2 chuzzle, ' : ''}row ${r + 1}, column ${c + 1}: ${COLORS[color]}`);
        frag.appendChild(d);
      }
    }
    el.replaceChildren(boardCanvas(el), frag);
  }

  // ---------------------------------------------------------------- chuzzles (the game's Chuzzle.cpp)
  // Pupil centres and travel per eye shape (gEyeCentres, gPupilRange), left eye then right.
  const EYE_PUPIL_CENTERS = [[[34, 60], [53, 62]], [[33, 62], [53, 63]], [[34, 63], [52, 62]], [[35, 62], [54, 62]]];
  const EYE_PUPIL_TRAVEL = [[4, 5], [5, 4], [4, 4], [5, 4]];
  // Eyes.png's top-left (85 x 88, drawn centred), in game pixels from the chuzzle's cell:
  // a chuzzle's face is centred at (40, 41) of its cell; the 2x2's at (84, 111) of its top-left
  // cell (Chuzzle::Draw: DrawEyes at (cx - 36, cy - 10) of the block's centre (80, 81)); and
  // on the palette's 2x2, whose sprite box is the 179-unit sprite drawn from (-9.5, -8.5).
  const EYES_AT = { normal: [-2, -3], piece: [42, 67], surface: [51.5, 75.5] };

  // Each chuzzle's own look, picked at random like the game's constructor: eye shape 0-3,
  // pupil colour 0-5, the face 0-2 pixels lower, a bounce speed of 0.2, 0.22 or 0.18 (/ 1.75)
  // per update at 100 updates a second over a 20-frame cycle, and a random start in it. Kept
  // per key (board and chuzzle id) so a chuzzle keeps its face as it moves.
  const looks = new Map();
  function look(key) {
    let f = looks.get(key);
    if (!f) {
      const speed = [0.2, 0.22, 0.18][Math.floor(Math.random() * 3)] / 1.75;
      f = { es: Math.floor(Math.random() * 4), ep: Math.floor(Math.random() * 6), oy: Math.floor(Math.random() * 3),
        sd: 20 / speed / 100, ph: Math.random() };
      looks.set(key, f);
    }
    return f;
  }

  function setChuzzle(el, color) {
    el.style.setProperty('--k', color);
  }

  const unit = (n) => `calc(${n} * var(--u))`;
  // Adds the eyes and pupils; key names the chuzzle (its look and eye state).
  function addFace(el, id, kind, key) {
    const f = look(key || `${kind}:${id}`);
    const [ex, ey0] = EYES_AT[kind];
    const ey = ey0 + f.oy;
    el.style.setProperty('--es', f.es);
    el.style.setProperty('--ep', f.ep);
    el.style.setProperty('--sd', `${f.sd.toFixed(3)}s`);
    el.style.setProperty('--ph', f.ph.toFixed(3));
    // the face: the eye shape and two pupils
    const face = document.createElement('i');
    face.className = 'face';
    const eyes = document.createElement('i');
    eyes.className = 'eyes';
    eyes.style.left = unit(ex);
    eyes.style.top = unit(ey);
    face.appendChild(eyes);
    for (const [cx, cy] of EYE_PUPIL_CENTERS[f.es]) {
      const pupil = document.createElement('i');
      pupil.className = 'pupil';
      pupil.style.left = unit(ex + cx - 7);
      pupil.style.top = unit(ey + cy - 7);
      pupil.style.setProperty('--tx', EYE_PUPIL_TRAVEL[f.es][0]);
      pupil.style.setProperty('--ty', EYE_PUPIL_TRAVEL[f.es][1]);
      face.appendChild(pupil);
    }
    el.appendChild(face);
    if (key) {
      el.dataset.eye = key;
      const s = eyeState(key);
      el.style.setProperty('--dx', s.dx);
      el.style.setProperty('--dy', s.dy);
    }
  }

  // Eye timers (chuzzle.py _update_eye_timers): a blink hides the pupils for 5 frames every
  // 3-10 s; the idle look direction changes every 5-14 s (a quarter of the time straight
  // ahead, otherwise a random angle at 0.35-1 of the travel); the pupils ease towards it.
  const eyes = new Map();
  const rand = (n) => Math.floor(Math.random() * n);
  function eyeState(key) {
    let s = eyes.get(key);
    if (!s) {
      const now = performance.now();
      s = { dx: 0, dy: 0, blinkAt: now + 3000 + rand(7000), lookAt: now + 300 + rand(1500) };
      eyes.set(key, s);
    }
    return s;
  }
  function randomLook() {
    if (Math.random() < 0.25) return [0, 0];
    const a = Math.random() * Math.PI * 2, r = 0.35 + Math.random() * 0.65;
    return [Math.cos(a) * r, -Math.sin(a) * r];
  }
  let tracked = null;           // the chuzzle the mouse is over (its eyes follow the mouse)
  let eyeEls = null;            // the faces on the page, found again after a redraw
  function eyeTick() {
    const now = performance.now();
    if (!eyeEls || !eyeEls.every((el) => el.isConnected)) eyeEls = [...document.querySelectorAll('[data-eye]:not(.cell)')];
    // Every eye change happens here, once per tick: each change makes the browser restyle the
    // page's animated parts, so a timer per blink (dozens a second) cost far more than the
    // animations. A blink lasts one tick (100 ms; the game's is 83). A chuzzle shown on both
    // boards blinks and looks on both.
    const blinks = new Set(), glances = new Set();
    for (const el of eyeEls) {
      const key = el.dataset.eye, s = eyeState(key);
      if (el.classList.contains('blink')) el.classList.remove('blink');
      else if (blinks.has(key) || now >= s.blinkAt) {
        if (!blinks.has(key)) s.blinkAt = now + 3000 + rand(7000);
        blinks.add(key);
        el.classList.add('blink');
      }
      if (el !== tracked && (glances.has(key) || now >= s.lookAt)) {
        if (!glances.has(key)) {
          s.lookAt = now + 5000 + rand(9000);
          [s.dx, s.dy] = randomLook();
        }
        glances.add(key);
        el.style.setProperty('--dx', s.dx.toFixed(3));
        el.style.setProperty('--dy', s.dy.toFixed(3));
      }
    }
  }
  // The mouse over a chuzzle: its pupils turn to the mouse (from the centre of its cell, or of
  // the 2x2's four); leaving holds that direction until the next idle look.
  let trackedKey = null;
  function trackMouse(e) {
    const t = document.elementFromPoint(e.clientX, e.clientY);
    const el = t && t.closest('.board .cell');
    if (tracked && tracked !== el) {
      eyeState(tracked.dataset.eye).lookAt = performance.now() + 5000 + rand(9000);
      tracked = null;
      trackedKey = null;
    }
    if (!el) return;
    const r = el.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    const len = Math.hypot(dx, dy);
    const s = eyeState(el.dataset.eye);
    [s.dx, s.dy] = len < 0.5 ? [0, 0] : [dx / len, dy / len];
    tracked = el;
    trackedKey = el.dataset.eye;
  }

  // ---------------------------------------------------------------- the boards' canvas
  // Each board's chuzzles are drawn on one canvas, the way a game draws: the cells under it are
  // see-through and only take the pointer (and show the pick and drop rings). A frame costs a
  // few hundred image copies and no restyling, whatever is moving.
  const SHEETS = {};
  for (const [name, file] of [['body', 'chuzzles'], ['huge', 'huge'], ['eyes', 'eyes'], ['pupils', 'pupils']]) {
    const img = new Image();
    img.src = `assets/${file}.webp`;
    img.decode().then(() => { SHEETS[name] = img; }, () => {});
  }

  function boardCanvas(el) {
    if (!el._cv) {
      el._cv = document.createElement('canvas');
      el._cv.className = 'board-canvas';
      el._cv.setAttribute('aria-hidden', 'true');
      el._ctx = el._cv.getContext('2d');
    }
    return el._cv;
  }

  // Where a chuzzle is in its bounce: the body frame and the face's bob (20 frames, 10 on
  // the 2x2; the face 0 or 1 game pixels lower, 0-2 on the 2x2).
  function bounce(f, piece, now) {
    if (reducedMotion.matches) return [0, 0];
    const p = (now / 1000 / f.sd + f.ph) % 1;
    if (piece) return [Math.floor(p * 10), p < 0.1 ? 0 : p < 0.3 ? 1 : p < 0.85 ? 2 : 1];
    return [Math.floor(p * 20), p >= 0.2 && p < 0.95 ? 1 : 0];
  }

  // The eye timers and the pupils' easing, once per chuzzle per frame (a chuzzle can be on
  // both boards).
  let eyeStamp = 0;
  function updateEyes(s, key, now, dt) {
    if (s.stamp === eyeStamp) return;
    s.stamp = eyeStamp;
    if (now >= s.blinkAt) {
      s.blinkUntil = now + 83;
      s.blinkAt = now + 83 + 3000 + rand(7000);
    }
    const track = trackedKey === key;
    if (!track && now >= s.lookAt) {
      s.lookAt = now + 5000 + rand(9000);
      [s.dx, s.dy] = randomLook();
    }
    // the idle look eases in over ~0.5 s, the mouse is followed at once
    const k = 1 - Math.exp(-dt / (track ? 25 : 120));
    s.px = (s.px ?? s.dx) + (s.dx - (s.px ?? s.dx)) * k;
    s.py = (s.py ?? s.dy) + (s.dy - (s.py ?? s.dy)) * k;
  }

  function drawBoard(el, now, dt) {
    const cv = boardCanvas(el), ctx = el._ctx;
    const u = layout.u, dpr = window.devicePixelRatio || 1, s = u * dpr;
    const size = Math.round(500 * s);
    if (cv.width !== size) {
      cv.width = size;
      cv.height = size;
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, size, size);
    if (!SHEETS.body || !SHEETS.huge || !SHEETS.eyes || !SHEETS.pupils) return;
    ctx.imageSmoothingQuality = 'high';
    const R = Math.round, cellPx = 80 * s, padPx = 10 * s;
    let lifted = null;
    const top = [];
    for (const node of el.children) {
      if (node._k === undefined) {
        if (!node.classList.contains('cell')) continue;
        node._k = +node.dataset.k;
        node._key = node.dataset.eye;
        node._piece = node.classList.contains('piece');
        node._r = +node.dataset.r;
        node._c = +node.dataset.c;
      }
      if (node._k !== node._k) continue;     // not a chuzzle
      if (node.classList.contains('lifted')) lifted = node;
      else if (node._sc > 1) top.push(node);
      else drawChuzzle(ctx, node, now, dt, s, R, cellPx, padPx, false);
    }
    for (const node of top) drawChuzzle(ctx, node, now, dt, s, R, cellPx, padPx, false);
    if (lifted) drawChuzzle(ctx, lifted, now, dt, s, R, cellPx, padPx, true);
  }

  function drawChuzzle(ctx, node, now, dt, s, R, cellPx, padPx, lifted) {
    const piece = node._piece, f = look(node._key), st = eyeState(node._key);
    updateEyes(st, node._key, now, dt);
    let frame, bob;
    if (node._f != null) [frame, bob] = [node._f, bounce(f, piece, now)[1]];
    else if (lifted || node.classList.contains('picked')) {
      if (node._hold == null) node._hold = bounce(f, piece, now);
      [frame, bob] = node._hold;
    } else {
      node._hold = null;
      [frame, bob] = bounce(f, piece, now);
    }
    // the cell's place, plus a lane slide (replay) or a drag (in CSS pixels, so times dpr)
    const dpr = s / layout.u;
    let x = padPx + node._c * cellPx, y = padPx + node._r * cellPx;
    if (node._off) {
      if (node._col) y += node._off * dpr;
      else x += node._off * dpr;
    }
    // easing (a swap's preview while dragging, a chuzzle settling into its cell after a drop
    // or swap): the offset _ox/_oy moves to _tx/_ty, the size _sc back to 1
    if (node._ox || node._oy || node._tx || node._ty) {
      const k = 1 - Math.exp(-dt / 55), tx = node._tx || 0, ty = node._ty || 0;
      node._ox = Math.abs(tx - (node._ox || 0)) < 0.2 ? tx : (node._ox || 0) + (tx - (node._ox || 0)) * k;
      node._oy = Math.abs(ty - (node._oy || 0)) < 0.2 ? ty : (node._oy || 0) + (ty - (node._oy || 0)) * k;
      x += node._ox * dpr;
      y += node._oy * dpr;
    }
    let sc = 1;
    if (lifted) {
      x += (node._dx || 0) * dpr;
      y += (node._dy || 0) * dpr;
      sc = 1.12;
    } else if (node._sc > 1) {
      node._sc = node._sc < 1.002 ? 1 : 1 + (node._sc - 1) * Math.exp(-dt / 55);
      sc = node._sc;
    }
    if (sc > 1) {
      const half = (piece ? 80 : 40) * s;
      ctx.save();
      ctx.translate(x + half, y + half);
      ctx.scale(sc, sc);
      ctx.translate(-x - half, -y - half);
      ctx.shadowColor = `rgba(0, 0, 0, ${(0.55 * (sc - 1) / 0.12).toFixed(3)})`;
      ctx.shadowBlur = 8 * dpr;
      ctx.shadowOffsetY = 6 * dpr;
    }
    if (piece) {
      ctx.drawImage(SHEETS.huge, frame * 180, node._k * 180, 179, 179, R(x - 9.5 * s), R(y - 8.5 * s), R(179 * s), R(179 * s));
    } else {
      ctx.drawImage(SHEETS.body, frame * 80, node._k * 80, 80, 80, R(x), R(y), R(cellPx), R(cellPx));
    }
    if (sc > 1) ctx.shadowColor = 'transparent';
    const [ex, ey0] = EYES_AT[piece ? 'piece' : 'normal'];
    const fx = x + ex * s, fy = y + (ey0 + f.oy + bob) * s;
    ctx.drawImage(SHEETS.eyes, f.es * 85, 0, 85, 88, R(fx), R(fy), R(85 * s), R(88 * s));
    if (now >= (st.blinkUntil || 0)) {
      const [tx, ty] = EYE_PUPIL_TRAVEL[f.es], ps = R(14 * s);
      for (const [cx, cy] of EYE_PUPIL_CENTERS[f.es]) {
        ctx.drawImage(SHEETS.pupils, f.ep * 14, 0, 14, 14,
          R(fx + (cx - 7 + st.px * tx) * s), R(fy + (cy - 7 + st.py * ty) * s), ps, ps);
      }
    }
    if (sc > 1) ctx.restore();
  }

  // Chuzzles easing into their cells after a board is redrawn: [r, c, dx, dy, size], the
  // offset (CSS pixels) and size each starts from.
  function glide(el, list) {
    for (const [r, c, dx, dy, sc] of list) {
      const n = el.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
      if (n) Object.assign(n, { _ox: dx, _oy: dy, _sc: sc || 1 });
    }
  }

  // One loop draws both boards every frame (a board out of sight, on a phone, is skipped).
  let lastFrame = 0;
  function frameLoop(now) {
    const dt = lastFrame ? Math.min(100, now - lastFrame) : 16;
    lastFrame = now;
    eyeStamp++;
    for (const id of ['goalBoard', 'startBoard']) {
      if (layout.two || state.side === (id === 'goalBoard' ? 'goal' : 'start')) drawBoard($(id), now, dt);
    }
    requestAnimationFrame(frameLoop);
  }


  // ---------------------------------------------------------------- the game's bitmap font
  // Bauhaus 93 from textures/fonts (metrics: tools/font/Bauhaus9336.py), drawn onto a canvas:
  // the glyphs filled with `fill`, a dark outline round them, and a soft shadow.
  const fontAtlas = new Image();
  const fontReady = new Promise((res) => { fontAtlas.onload = res; fontAtlas.onerror = res; });
  fontAtlas.src = 'assets/bauhaus36.webp';
  const FONT = self.MB_FONT;
  const fontJobs = new Map();
  function fontText(el, text, { size = 18, fill = ['#ffffff', '#e8e4ff'], outline = 'rgba(30, 10, 50, .95)' } = {}) {
    fontJobs.set(el, { text, size, fill, outline });
    el.classList.add('ftext');
    el.setAttribute('aria-label', text);
    drawFontText(el);
  }
  function drawFontText(el) {
    const job = fontJobs.get(el);
    if (!job || !FONT || !fontAtlas.naturalWidth) return;
    const { text, size, fill, outline } = job;
    const s = size / FONT.size, dpr = Math.min(3, window.devicePixelRatio || 1);
    const o = Math.max(1, Math.round(size / 9));                       // outline width, CSS px
    let w = 0;
    for (const ch of text) w += ch === ' ' ? FONT.space : (FONT.glyphs[ch] ? FONT.glyphs[ch].w : FONT.space);
    const cw = Math.ceil(w * s + 2 * o + 2), ch = Math.ceil(FONT.height * s + 2 * o + 2);
    let cv = el.querySelector('canvas');
    if (!cv) {
      cv = document.createElement('canvas');
      cv.setAttribute('aria-hidden', 'true');
      el.replaceChildren(cv);
    }
    cv.width = cw * dpr;
    cv.height = ch * dpr;
    cv.style.width = `${cw}px`;
    cv.style.height = `${ch}px`;
    const glyphs = document.createElement('canvas');
    glyphs.width = cv.width;
    glyphs.height = cv.height;
    const g = glyphs.getContext('2d');
    g.scale(dpr, dpr);
    let x = o + 1;
    for (const c of text) {
      const gl = FONT.glyphs[c];
      if (gl) {
        const [rx, ry, rw, rh] = gl.r;
        g.drawImage(fontAtlas, rx, ry, rw, rh, x + gl.o[0] * s, o + 1 + gl.o[1] * s, rw * s, rh * s);
      }
      x += (gl ? gl.w : FONT.space) * s;
    }
    const tint = (style) => {
      const c = document.createElement('canvas');
      c.width = glyphs.width;
      c.height = glyphs.height;
      const k = c.getContext('2d');
      k.drawImage(glyphs, 0, 0);
      k.globalCompositeOperation = 'source-in';
      k.fillStyle = style;
      k.fillRect(0, 0, c.width, c.height);
      return c;
    };
    const ctx = cv.getContext('2d');
    ctx.clearRect(0, 0, cv.width, cv.height);
    const grad = ctx.createLinearGradient(0, (o + 1 + 8 * s) * dpr, 0, (o + 1 + FONT.ascent * s) * dpr);
    grad.addColorStop(0, fill[0]);
    grad.addColorStop(1, fill[1]);
    const dark = tint(outline);
    ctx.shadowColor = 'rgba(0, 0, 0, .55)';
    ctx.shadowBlur = 4 * dpr;
    ctx.shadowOffsetY = 1.5 * dpr;
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * Math.PI * 2;
      ctx.drawImage(dark, Math.cos(a) * o * dpr, Math.sin(a) * o * dpr);
      if (k === 0) ctx.shadowColor = 'transparent';
    }
    ctx.drawImage(tint(grad), 0, 0);
  }
  const redrawFonts = () => { for (const el of fontJobs.keys()) if (el.isConnected) drawFontText(el); };
  fontReady.then(redrawFonts);
  const GOLD_TEXT = { size: 20, fill: ['#fff6b8', '#ffb71e'] };
  const LABEL_TEXT = { size: 14 };                                   // under the dock's buttons
  const ORB_TEXT = { size: 22, fill: ['#ffffff', '#ffe9a8'], outline: 'rgba(60, 10, 0, .95)' };
  const captionStyle = () => ({ ...GOLD_TEXT, size: Math.round(Math.max(18, Math.min(30, 34 * layout.u))) });
  const infoSize = () => Math.round(Math.max(15, Math.min(22, innerWidth / 42)));

  // ---------------------------------------------------------------- layout
  // The boards fill the space between the info bar and the dock: two side by side when that
  // gives at least ~3/4 of the size one board would get, else one with Start / Goal tabs.
  // --u (one game pixel) follows; a board is 535 of them.
  const layout = { u: 0.6, two: false };
  function layoutStage() {
    const st = $('stage');
    const W = st.clientWidth, H = st.clientHeight;
    if (!W || !H) return;
    const pad = 16, caption = 36, tabs = 56;
    const arrow = Math.max(46, Math.min(72, 64 * layout.u)) + 36;
    const u2 = Math.min((W - pad - arrow) / 1070, (H - pad - caption) / 535);
    const u1 = Math.min((W - pad) / 535, (H - pad - tabs) / 535);
    const two = u2 >= u1 * 0.72;
    // Snapped so 20 game pixels are a whole number of screen pixels: then every frame of the
    // sprite strips (80 and 180 units) starts on a pixel, and the chuzzles sway in place
    // instead of jittering sideways as the frame offsets round differently.
    const dpr = window.devicePixelRatio || 1;
    const u = Math.floor(Math.max(0.25, Math.min(1.6, two ? u2 : u1)) * 20 * dpr) / (20 * dpr);
    snapPalette(dpr);
    const moved = Math.abs(u - layout.u) > 0.002 || two !== layout.two;
    layout.u = u;
    layout.two = two;
    st.classList.toggle('two', two);
    st.classList.toggle('one', !two);
    document.documentElement.style.setProperty('--u', `${u.toFixed(4)}px`);
    if (moved) {
      fontText($('startCap'), state.mode === 'replay' ? 'Replay' : 'Start', captionStyle());
      fontText($('goalCap'), 'Goal', captionStyle());
      if (state.mode === 'replay' && result && result.replay && !result.playing) showStep(result.step);
    }
  }

  // The palette's chuzzles are sprites too: their size (and the 2x2's) in whole screen pixels.
  function snapPalette(dpr) {
    const pal = $('palette');
    pal.style.removeProperty('--sz');
    pal.style.removeProperty('--pg');
    const s = pal.querySelector('.swatch:not(.big)');
    if (!s) return;
    const px = (v) => `${Math.max(1, Math.floor(v * dpr)) / dpr}px`;
    pal.style.setProperty('--sz', px(parseFloat(getComputedStyle(s).width)));
    pal.style.setProperty('--pg', px(parseFloat(getComputedStyle(pal).getPropertyValue('--pg')) || 10));
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
    const u = pitch / 80;
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
    for (const [node] of moving) node.classList.add('mv');
    // chuzzle.py _get_frame_surface: frame = sway count + direction x speed count + speed
    // index, the speed index from sqrt(speed / (10% of a cell per frame)), the 2x2 at half speed
    let lastP = 0, lastT = performance.now(), smooth = 0;
    const frameFor = (piece) => {
      const t = Math.sqrt(Math.min(1, (piece ? smooth * 0.5 : smooth) / 8));
      return piece ? 10 + seg * 5 + Math.round(t * 4) : 20 + seg * 10 + Math.round(t * 9);
    };
    const set = (p) => {
      const now = performance.now();
      const frames = Math.max(0.25, (now - lastT) / (1000 / 60));
      const speed = Math.abs(p - lastP) * Math.abs(d) * 80 / frames;  // game pixels per frame
      smooth = smooth * 0.5 + speed * 0.5;
      lastP = p;
      lastT = now;
      for (const [node, a, b] of moving) {
        node._off = (a + (b - a) * p) * pitch;
        node._col = mv.col;
        node._f = frameFor(node.classList.contains('piece'));
      }
    };
    set(0);
    return {
      set,
      finish() {
        el.querySelectorAll('.ghost').forEach((g) => g.remove());
        for (const [node] of moving) {
          node._off = 0;
          node._f = null;
          node.classList.remove('mv');
        }
      },
    };
  }

  // The tab's little board: a dot per chuzzle, the 2x2 as one big dot over its four cells.
  function renderThumb(el, board) {
    const frag = document.createDocumentFragment();
    const f = board.fat;
    board.cells.forEach((v, k) => {
      const r = Math.floor(k / 6), c = k % 6;
      const piece = inPiece(f, r, c);
      if (piece && (r !== f.y || c !== f.x)) return;
      const i = document.createElement('i');
      i.style.setProperty('--col', `var(--c${v})`);
      i.style.gridArea = `${r + 1} / ${c + 1} / span ${piece ? 2 : 1} / span ${piece ? 2 : 1}`;
      if (piece) i.className = 'p';
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
        setChuzzle(s, k);
        addFace(s, k * 3 + 1, 'normal', `sw:${k}`);
        s.insertAdjacentHTML('beforeend', '<span class="n"></span>');
        s.addEventListener('click', () => selectColour(k));
        el.appendChild(s);
      });
      const big = document.createElement('button');
      big.type = 'button';
      big.className = 'swatch big';
      big.id = 'fatSwatch';
      big.setAttribute('role', 'radio');
      addFace(big, 40, 'surface', 'sw:big');
      big.insertAdjacentHTML('beforeend', '<span class="n">2x2</span>');
      big.addEventListener('click', selectFatTool);
      el.appendChild(big);
    }
    COLORS.forEach((name, k) => {
      const s = el.children[k];
      const ok = a[k] === b[k];
      s.setAttribute('aria-checked', String(state.tool === 'paint' && state.color === k));
      s.setAttribute('aria-label', `${name}: ${a[k]} on start, ${b[k]} on goal`);
      s.title = `${name} (key ${k + 1})${ok ? '' : ` - ${a[k]} on start, ${b[k]} on goal`}`;
      const n = s.querySelector('.n');
      n.className = ok ? 'n' : 'n bad';
      n.textContent = a[k] || b[k] ? (ok ? a[k] : `${a[k]}≠${b[k]}`) : '';
    });
    const big = $('fatSwatch');
    setChuzzle(big, pieceColour(state.goal) ?? state.color);
    big.setAttribute('aria-checked', String(state.tool === 'piece'));
    big.setAttribute('aria-label', '2x2 Chuzzle: tap a cell to place it');
    big.title = state.goal.fat ? '2x2 Chuzzle: tap the goal to move it (key 9)' : '2x2 Chuzzle: tap the goal to place it (key 9)';
  }

  function render() {
    const replay = state.mode === 'replay' && result && result.replay;
    for (const wrap of document.querySelectorAll('.board-wrap')) {
      wrap.classList.toggle('active', wrap.dataset.side === state.side);
    }
    if (!replay) renderBoard($('startBoard'), state.start);
    renderBoard($('goalBoard'), state.goal);
    const game = !!state.preset;
    for (const side of ['start', 'goal']) {
      const b = $(`${side}Board`);
      const edit = !replay && !game;
      b.classList.toggle('placing', edit && side === 'goal' && state.tool === 'piece');
      b.classList.toggle('readonly', !edit);
      b.classList.toggle('mover', edit && side === 'start');
    }
    $('customBtn').setAttribute('aria-pressed', String(!game));
    $('puzzlesBtn').setAttribute('aria-pressed', String(game));
    for (const [id, on] of [['customBtn', !game], ['puzzlesBtn', game]]) {
      const label = $(id).querySelector('[data-label]');
      fontText(label, label.dataset.label, on ? { ...GOLD_TEXT, size: 16 } : { size: 16, fill: ['#d8d2f4', '#a9a2cf'] });
    }
    renderThumb($('thumbStart'), replay ? result.replay[result.step] : state.start);
    renderThumb($('thumbGoal'), state.goal);
    $('tabStart').setAttribute('aria-selected', String(state.side === 'start'));
    $('tabGoal').setAttribute('aria-selected', String(state.side === 'goal'));
    const tab = (on) => (on ? { ...GOLD_TEXT, size: 19 } : { size: 19, fill: ['#d8d2f4', '#a9a2cf'] });
    fontText($('tabStartLabel'), replay ? 'Replay' : 'Start', tab(state.side === 'start'));
    fontText($('tabGoalLabel'), 'Goal', tab(state.side === 'goal'));
    fontText($('startCap'), replay ? 'Replay' : 'Start', captionStyle());
    fontText($('goalCap'), 'Goal', captionStyle());
    $('editDock').hidden = !!replay || game;
    $('knownNote').hidden = !game || !!replay;
    if (game) {
      const p = presetByName(state.preset);
      $('knownNote').textContent = `${p.solutions.toLocaleString()} shortest solution${p.solutions > 1 ? 's' : ''} of ${p.moves} moves`;
    }
    $('replayDock').hidden = !replay;
    $('solSwitch').hidden = !replay;
    $('info').hidden = !!replay || !!state.coach;
    $('guide').hidden = !replay;
    $('unfatBtn').disabled = !state.goal.fat;
    $('resetBtn').disabled = state.follows;
    renderDepth();
    $('fillBtn').title = `Fill the goal with ${COLORS[state.color].toLowerCase()}`;
    $('undoBtn').disabled = state.undo.length === 0;
    renderPalette();
    if (!run) {
      fontText($('solveLabel'), game ? 'View' : 'Solve', ORB_TEXT);
      $('solveBtn').setAttribute('aria-label', game ? 'View the solutions' : 'Solve');
    }
    if (replay) showStep(result.step);
  }

  // Colours and the 2x2 paint the goal (the board shown on phones switches to it).
  function selectColour(k) {
    state.color = k;
    state.tool = 'paint';
    if (state.mode === 'edit') state.side = 'goal';
    render();
  }

  function selectFatTool() {
    state.tool = state.tool === 'piece' ? 'paint' : 'piece';
    if (state.mode === 'edit') state.side = 'goal';
    render();
  }

  function setSide(side) {
    state.side = side;
    render();
  }

  // The move limit beside Solve ("up to / 10 moves", with - and +). A game puzzle longer than
  // the page's search has no limit to set: Solve shows the desktop solver's answer.
  function renderDepth() {
    const d = state.maxDepth;
    $('depthBox').hidden = !!knownPreset();
    fontText($('depthUp'), 'up to', { size: 12, fill: ['#e6e0ff', '#bdb7de'] });
    fontText($('depthText'), `${d} move${d > 1 ? 's' : ''}`, { size: 17 });
    $('depthDown').disabled = d <= 1;
    $('depthUpBtn').disabled = d >= MAX_DEPTH;
  }

  function setDepth(d) {
    state.maxDepth = Math.max(1, Math.min(MAX_DEPTH, d));
    renderDepth();
  }

  // ---------------------------------------------------------------- editing
  // The goal is painted (the selected colour, or the 2x2: tap where its top-left goes). The
  // start's chuzzles are dragged: drop one on another and they swap, or tap two; the 2x2 is
  // dragged to its spot and the chuzzles it covers take the place it leaves.
  function attachEditor(el, which) {
    let g = null;              // the gesture under way
    const cellAt = (e) => {
      const t = document.elementFromPoint(e.clientX, e.clientY);
      const cell = t && t.closest('.cell');
      return cell && el.contains(cell) && !cell.classList.contains('ghost') ? cell : null;
    };
    const editable = () => state.mode === 'edit' && !state.preset;
    const paintAt = (cell) => {
      const r = +cell.dataset.r, c = +cell.dataset.c;
      const b = state.goal;
      const before = b.cells.join() + JSON.stringify(b.fat);
      if (state.tool === 'piece') placeFat('goal', r, c);
      else paint(b, r, c, state.color);
      if (b.cells.join() + JSON.stringify(b.fat) === before) return;
      renderBoard(el, b);
      syncStart();
      renderBoard($('startBoard'), state.start);
    };

    // The cell under a point (row, column), from the board's geometry.
    const spotAt = (x, y) => {
      const b = el.getBoundingClientRect(), pitch = 80 * layout.u;
      const c = Math.floor((x - b.left - 10 * layout.u) / pitch), r = Math.floor((y - b.top - 10 * layout.u) / pitch);
      return r >= 0 && r < 6 && c >= 0 && c < 6 ? { r, c } : null;
    };
    // Where a drag would drop: a cell for a chuzzle, a top-left for the 2x2 (it moves by the
    // drag, kept inside the board). null where nothing would change.
    const dropTarget = (e) => {
      const piece = g.cell.classList.contains('piece');
      if (piece) {
        const pitch = 80 * layout.u;
        const r = Math.max(0, Math.min(4, g.r + Math.round((e.clientY - g.y) / pitch)));
        const c = Math.max(0, Math.min(4, g.c + Math.round((e.clientX - g.x) / pitch)));
        return r === g.r && c === g.c ? null : { r, c, piece };
      }
      const s = spotAt(e.clientX, e.clientY);
      if (!s || (s.r === g.r && s.c === g.c) || inPiece(state[which].fat, s.r, s.c)) return null;
      return { ...s, piece };
    };
    const showTarget = (t) => {
      const old = el.querySelector('.cell.drop-on');
      if (old) old.classList.remove('drop-on');
      if (t && !t.piece) {
        const cell = el.querySelector(`.cell[data-r="${t.r}"][data-c="${t.c}"]`);
        if (cell) cell.classList.add('drop-on');
      }
    };

    el.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !editable()) return;
      const cell = cellAt(e);
      if (!cell) return;
      e.preventDefault();
      try { el.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
      if (state.side !== which) {
        state.side = which;
        for (const wrap of document.querySelectorAll('.board-wrap')) wrap.classList.toggle('active', wrap.dataset.side === which);
      }
      g = { x: e.clientX, y: e.clientY, r: +cell.dataset.r, c: +cell.dataset.c, cell, move: which === 'start' };
      if (!g.move) {
        pushUndo();
        paintAt(cell);
      }
    });
    el.addEventListener('pointermove', (e) => {
      if (!g) return;
      if (!g.move) {
        if (state.tool === 'piece') return;
        // paint every cell the stroke crosses, not only where pointer events landed (a fast
        // drag reports few points)
        const lx = g.lx ?? g.x, ly = g.ly ?? g.y;
        const steps = Math.max(1, Math.ceil(Math.hypot(e.clientX - lx, e.clientY - ly) / (20 * layout.u)));
        for (let k = 1; k <= steps; k++) {
          const cell = cellAt({ clientX: lx + (e.clientX - lx) * k / steps, clientY: ly + (e.clientY - ly) * k / steps });
          if (cell) paintAt(cell);
        }
        g.lx = e.clientX;
        g.ly = e.clientY;
        return;
      }
      const dx = e.clientX - g.x, dy = e.clientY - g.y;
      if (!g.drag && Math.hypot(dx, dy) > 6) {
        g.drag = true;
        g.cell.classList.add('lifted');
        el.classList.add('dragging');
        clearPick();
      }
      if (g.drag) {
        g.cell._dx = dx;
        g.cell._dy = dy;
        const t = dropTarget(e);
        if (!g.target !== !t || (t && (t.r !== g.target.r || t.c !== g.target.c))) {
          showTarget(t);
          if (g.preview) Object.assign(g.preview, { _tx: 0, _ty: 0 });
          g.preview = null;
          if (t && !t.piece) {
            const n = el.querySelector(`.cell[data-r="${t.r}"][data-c="${t.c}"]`);
            const P = 80 * layout.u;
            if (n) Object.assign(n, { _tx: (g.c - t.c) * P, _ty: (g.r - t.r) * P });
            g.preview = n;
          }
        }
        g.target = t;
      }
    });
    const end = (e) => {
      if (!g) return;
      const gesture = g;
      g = null;
      if (!gesture.move) {
        changed();
        return;
      }
      if (gesture.drag) {
        el.classList.remove('dragging');
        if (e.type === 'pointerup' && gesture.target) dropOn(which, gesture, gesture.target);
        else {
          renderBoard(el, state[which]);
          glide(el, [[gesture.r, gesture.c, gesture.cell._dx || 0, gesture.cell._dy || 0, 1.12]]);
        }
        return;
      }
      if (e.type === 'pointerup') tapOn(which, gesture.cell);
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('contextmenu', (e) => {
      const cell = cellAt(e);
      if (!cell) return;
      e.preventDefault();
      const b = which === 'start' && state.mode === 'replay' ? result.replay[result.step] : state[which];
      selectColour(b.cells[+cell.dataset.r * 6 + +cell.dataset.c]);
    });
  }

  function clearPick() {
    for (const c of document.querySelectorAll('.cell.picked')) c.classList.remove('picked');
    state.picked = null;
  }

  // The start rearranged.
  function moved(which, b) {
    pushUndo();
    state[which] = b;
    state.follows = b.ids.every((v, k) => v === k);
    clearPick();
    changed();
  }

  // A drop: a chuzzle swaps with the one where it lands; the 2x2 moves to its new top-left,
  // and the chuzzles it now covers go to the cells it left (in reading order).
  function dropOn(which, from, to) {
    const old = state[which];
    const b = cloneBoard(old);
    const ids = old.ids || old.cells.map((_, k) => k);
    b.ids = ids.slice();
    const el = $(`${which}Board`), P = 80 * layout.u;
    const glides = [[to.r, to.c, (from.c - to.c) * P + (from.cell._dx || 0), (from.r - to.r) * P + (from.cell._dy || 0), 1.12]];
    if (to.piece) {
      const nf = { x: to.c, y: to.r };
      const was = pieceCells(old.fat), now = pieceCells(nf);
      const freed = was.filter((k) => !now.includes(k)), taken = now.filter((k) => !was.includes(k));
      freed.forEach((k, n) => {
        b.cells[k] = old.cells[taken[n]];
        b.ids[k] = ids[taken[n]];
        glides.push([Math.floor(k / 6), k % 6, (taken[n] % 6 - k % 6) * P, (Math.floor(taken[n] / 6) - Math.floor(k / 6)) * P]);
      });
      now.forEach((k, m) => {
        b.cells[k] = old.cells[was[m]];
        b.ids[k] = ids[was[m]];
      });
      b.fat = nf;
    } else {
      const i = from.r * 6 + from.c, j = to.r * 6 + to.c;
      [b.cells[i], b.cells[j]] = [b.cells[j], b.cells[i]];
      [b.ids[i], b.ids[j]] = [b.ids[j], b.ids[i]];
      // the swapped one was already sliding over (the preview): it carries on from there
      const n = el.querySelector(`.cell[data-r="${to.r}"][data-c="${to.c}"]`);
      glides.push([from.r, from.c, (to.c - from.c) * P + ((n && n._ox) || 0), (to.r - from.r) * P + ((n && n._oy) || 0)]);
    }
    moved(which, b);
    glide(el, glides);
  }

  // A tap on the start: the first chuzzle is picked, a second one swaps with it.
  // The 2x2 is dragged, never swapped.
  function tapOn(which, cell) {
    const i = +cell.dataset.r * 6 + +cell.dataset.c;
    const p = state.picked;
    clearPick();
    if (cell.classList.contains('piece') || (p && p.side === which && p.i === i)) return;
    if (!p || p.side !== which) {
      state.picked = { side: which, i };
      cell.classList.add('picked');
      return;
    }
    const b = cloneBoard(state[which]);
    if (!b.ids) b.ids = b.cells.map((_, k) => k);
    [b.cells[i], b.cells[p.i]] = [b.cells[p.i], b.cells[i]];
    [b.ids[i], b.ids[p.i]] = [b.ids[p.i], b.ids[i]];
    moved(which, b);
    const P = 80 * layout.u, r1 = Math.floor(i / 6), c1 = i % 6, r2 = Math.floor(p.i / 6), c2 = p.i % 6;
    glide($(`${which}Board`), [[r1, c1, (c2 - c1) * P, (r2 - r1) * P], [r2, c2, (c1 - c2) * P, (r1 - r2) * P]]);
  }

  function changed({ keepPreset = false } = {}) {
    if (!state.preset && !state.start.ids) syncStart();   // a custom start is the goal rearranged
    if (!keepPreset && state.preset) {
      const p = presetByName(state.preset);
      if (!p || !sameBoard(state.start, presetBoard(p, 'start')) || !sameBoard(state.goal, presetBoard(p, 'goal'))) {
        state.preset = '';
      }
    }
    render();
    renderInfo();
    writeHash();
    markStale();
    if (!run) {
      showError('');
      idleStatus();
    }
  }

  function removeFat() {
    pushUndo();
    state.start.fat = null;
    state.goal.fat = null;
    if (state.tool === 'piece') state.tool = 'paint';
    syncStart();
    changed();
  }

  function undo() {
    const s = state.undo.pop();
    if (!s) return;
    state.start = s.start;
    state.goal = s.goal;
    state.preset = s.preset;
    state.follows = s.follows;
    state.picked = null;
    leaveReplay();
    changed({ keepPreset: true });
  }

  // ---------------------------------------------------------------- the tutorial
  // Three steps in the info bar, each lighting what it talks about: paint the goal; drag the
  // start's chuzzles to where they begin (or Mix); Solve. Shown on a first visit.
  const COACH = [
    null,
    { side: 'goal', text: '<b>Paint the goal:</b> pick a colour below, then click or drag on the Goal board.', lit: ['goalWrap', 'palette'] },
    { side: 'start', text: '<b>Now the start:</b> drag its Chuzzles to where they begin (they swap places), or press <b>Mix</b>.', lit: ['startWrap', 'scrambleBtn'] },
    { side: null, text: '<b>Press Solve</b> to find every shortest solution.', lit: ['solveBtn'] },
  ];
  function coachStep(n) {
    for (const el of document.querySelectorAll('.lit')) el.classList.remove('lit');
    if (n >= COACH.length) {
      endCoach(true);
      return;
    }
    state.coach = n;
    const s = COACH[n];
    $('coach').hidden = false;
    $('coachN').textContent = n;
    $('coachText').innerHTML = s.text;
    $('coachNext').textContent = n === COACH.length - 1 ? 'Got it' : 'Next';
    $('coachSkip').hidden = n === COACH.length - 1;
    for (const id of s.lit) $(id).classList.add('lit');
    if (s.side) state.side = s.side;
    render();
    renderInfo();
  }
  function endCoach(done = true) {
    if (!state.coach) return;
    state.coach = 0;
    for (const el of document.querySelectorAll('.lit')) el.classList.remove('lit');
    $('coach').hidden = true;
    if (done) try { localStorage.setItem('mb-tutorial', 'done'); } catch { /* storage blocked */ }
    renderInfo();
  }

  // ---------------------------------------------------------------- presets and links
  const PUZZLES = self.MB_PUZZLES || [];
  const presetByName = (name) => PUZZLES.find((p) => p.name === name);
  // A game puzzle open (boards as the game has them): all 100 are solved, and their shortest
  // solutions are in solutions/<name>.txt, so it is viewed, not edited or searched.
  const knownPreset = () => (state.preset && presetByName(state.preset)) || null;
  const presetBoard = (p, side) => ({
    cells: p[side].slice(),
    fat: p.fat ? { x: p.fat[side][0], y: p.fat[side][1] } : null,
  });

  // The game's puzzles, a row per world, in their own dialog: painting your own board is
  // what the page is for, so they stay out of the way until asked for.
  function fillPresets() {
    const grid = $('pzGrid');
    let row = null, world = '';
    for (const p of PUZZLES) {
      const w = p.name.split('-')[0];
      if (w !== world) {
        world = w;
        row = document.createElement('div');
        row.className = 'pz-row';
        row.innerHTML = `<span class="pz-world">${w}</span>`;
        grid.appendChild(row);
      }
      const b = document.createElement('button');
      b.type = 'button';
      b.className = p.fat ? 'pz fat' : 'pz';
      b.dataset.name = p.name;
      b.title = `Puzzle ${p.name}: ${p.moves} moves, ${p.solutions.toLocaleString()} solution${p.solutions > 1 ? 's' : ''}${p.fat ? ', with the 2x2' : ''}`;
      b.innerHTML = `<b>${p.name}</b><span>${p.moves}</span>`;
      row.appendChild(b);
    }
    grid.addEventListener('click', (e) => {
      const b = e.target.closest('.pz');
      if (!b) return;
      $('puzzlesDialog').close();
      loadPreset(b.dataset.name);
    });
  }

  function loadPreset(name) {
    const p = presetByName(name);
    if (!p) return;
    if (!state.preset) state.custom = { start: cloneBoard(state.start), goal: cloneBoard(state.goal), follows: state.follows };
    endCoach();
    pushUndo();
    state.start = presetBoard(p, 'start');
    state.goal = presetBoard(p, 'goal');
    state.preset = name;
    state.follows = false;
    state.picked = null;
    leaveReplay();
    changed({ keepPreset: true });
  }

  // Back to the custom puzzle (the one left for a game puzzle, else a blank goal to paint).
  function enterCustom() {
    if (!state.preset) return;
    pushUndo();
    const c = state.custom;
    state.start = c ? c.start : blank();
    state.goal = c ? c.goal : blank();
    state.follows = c ? c.follows : true;
    state.preset = '';
    state.picked = null;
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
    let text, warn = false;
    $('info').hidden = !!state.coach || state.mode === 'replay';
    if (startMismatch()) {
      text = 'Start and goal need the same Chuzzles: the red numbers show which';
      warn = true;
    } else if (!p) {
      text = state.follows ? 'Paint the goal, then drag the start\'s Chuzzles into place' : state.start.fat ? 'Your puzzle, with the 2x2 Chuzzle' : 'Your puzzle';
    } else if (p.moves) {
      text = `Puzzle ${p.name}  -  shortest is ${p.moves} moves  -  ${p.solutions.toLocaleString()} solution${p.solutions > 1 ? 's' : ''}`;
    } else {
      text = `Puzzle ${p.name} is unsolved: it needs at least ${p.atLeast} moves`;
      warn = true;
    }
    fontText(el, text, warn ? { size: infoSize(), fill: ['#ffd2c8', '#ff8f7a'] } : { size: infoSize() });
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
      state.follows = sameBoard(state.start, state.goal);
      if (state.follows) state.start = copyOfGoal();
      else linkStart();
      return true;
    }
    const p = presetByName(q.get('p'));
    if (p) {
      state.start = presetBoard(p, 'start');
      state.goal = presetBoard(p, 'goal');
      state.preset = p.name;
      state.follows = false;
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

  const fmtTime = (s) => (s < 0.1 ? 'under 0.1 s' : s < 10 ? `${s.toFixed(1)} s` : s < 90 ? `${Math.round(s)} s`
    : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`);

  // The Solve orb: "Solve" when idle; while a solve runs it fills from the bottom with the
  // progress (or bobs when the step has none) and shows the percentage; `label` is what the
  // status line says the solver is doing.
  let stage = '';
  function setProgress(label, fraction) {
    const idle = !run;
    stage = idle ? '' : label;
    fontText($('solveLabel'), idle ? (state.preset ? 'View' : 'Solve') : fraction == null ? '...' : `${Math.floor(fraction * 100)}%`, ORB_TEXT);
    const fill = $('solveFill');
    fill.classList.toggle('indet', !idle && fraction == null);
    fill.style.transform = idle || fraction == null ? '' : `scaleY(${Math.round(fraction * 1000) / 1000})`;
  }

  // The status line when nothing runs: the last solve's summary while the boards are still
  // the ones it solved.
  function idleStatus() {
    const el = $('status');
    if (run || el.classList.contains('error')) return;
    const same = result && sameBoard(result.boards.start, state.start) && sameBoard(result.boards.goal, state.goal);
    el.textContent = same ? result.summary : '';
    el.title = el.textContent;
  }

  function showError(text) {
    const el = $('status');
    if (text) {
      el.textContent = text;
      el.title = text;
      el.classList.add('error');
    } else if (el.classList.contains('error')) {
      el.textContent = '';
      el.classList.remove('error');
    }
  }

  function problemSpec() {
    const a = colourCounts(state.start), b = colourCounts(state.goal);
    const diff = [];
    for (let k = 0; k < 8; k++) if (a[k] !== b[k]) diff.push(`${COLORS[k].toLowerCase()} ${a[k]} vs ${b[k]}`);
    if (diff.length || !!state.start.fat !== !!state.goal.fat) {
      return { error: 'The start and the goal need the same Chuzzles: press Reset to copy the goal to the start again.' };
    }
    const fat = state.start.fat
      ? { start: [state.start.fat.x, state.start.fat.y], goal: [state.goal.fat.x, state.goal.fat.y] } : null;
    return { spec: { start: state.start.cells.slice(), goal: state.goal.cells.slice(), fat } };
  }

  async function solve() {
    if (run) return;
    if (state.coach) endCoach(true);
    const { spec, error } = problemSpec();
    if (error) {
      showError(error);
      return;
    }
    showError('');
    const known = knownPreset();
    const maxDepth = state.maxDepth;
    const threads = +$('threads').value || hardwareThreads;
    run = { workers: [], cancelled: false, t0: performance.now(), boards: { start: cloneBoard(state.start), goal: cloneBoard(state.goal) } };
    const thisRun = run;
    $('solveBtn').classList.add('running');
    $('solveBtn').disabled = true;
    $('cancelBtn').hidden = false;
    $('depthBox').classList.add('busy');
    let detail = '';
    const showStatus = () => {
      $('status').textContent = `${stage} · ${fmtTime((performance.now() - thisRun.t0) / 1000)}${detail}`;
    };
    const tick = setInterval(showStatus, 200);
    try {
      if (known) {
        setProgress('Loading', null);
        showStatus();
        const r = await fetch(`solutions/${known.name}.txt`);
        if (!r.ok) throw new Error(`could not load solutions/${known.name}.txt (${r.status})`);
        const text = await r.text();
        if (thisRun.cancelled) return;
        const count = text.split('\n').filter(Boolean).length;
        finish(thisRun, { depth: known.moves, count, text, threads: 0, maxDepth: known.moves, known: true });
        return;
      }
      await workerUrlReady;
      const w0 = warm || new SolverWorker();
      warm = null;
      run.workers.push(w0);
      setProgress('Starting', null);
      showStatus();
      await w0.call('setup', spec);
      setProgress(`Up to ${Math.min(6, maxDepth)} moves`, null);
      const small = await w0.call('small', { maxDepth });
      if (small.depth >= 0) {
        finish(thisRun, { depth: small.depth, count: small.count, text: small.text, threads: 1, maxDepth });
        return;
      }
      if (maxDepth <= 6) {
        finish(thisRun, { depth: -1, threads: 1, maxDepth });
        return;
      }
      setProgress('Building tables', null);
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
        setProgress(`${d} moves`, 0);
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
      $('depthBox').classList.remove('busy');
      setProgress('', 0);
      if (thisRun.cancelled) {
        if (!$('status').classList.contains('error')) $('status').textContent = 'Cancelled';
      } else {
        idleStatus();
      }
      warmUp();
    }
  }

  function cancel() {
    if (!run) return;
    run.cancelled = true;
    for (const w of run.workers) w.terminate();
  }

  // ---------------------------------------------------------------- results
  function finish(r, res) {
    if (r.cancelled) return;
    const secs = (performance.now() - r.t0) / 1000;
    const lines = res.text ? res.text.split('\n').filter(Boolean).sort() : [];
    const threads = `${res.threads} thread${res.threads > 1 ? 's' : ''}`;
    const summary = res.known ? '' : `${res.depth < 0 ? 'Searched' : 'Solved'} in ${fmtTime(secs)} on ${threads}`;
    result = { ...res, lines, secs, summary, boards: r.boards, shown: 0, selected: -1, step: 0, playing: false };
    $('results').hidden = false;
    if (res.depth < 0) {
      fontText($('resultTitle'), `No solution in ${res.maxDepth} moves`, { size: 24, fill: ['#fff6b8', '#ffb71e'] });
      $('resultSub').textContent = `Every sequence of up to ${res.maxDepth} moves was checked.`;
    } else if (res.depth === 0) {
      fontText($('resultTitle'), 'Already solved', { size: 24, fill: ['#fff6b8', '#ffb71e'] });
      $('resultSub').textContent = 'The start board is the goal board.';
    } else {
      const n = res.count;
      fontText($('resultTitle'), `${res.depth} move${res.depth > 1 ? 's' : ''}`, { size: 30, fill: ['#fff6b8', '#ffb71e'] });
      $('resultSub').textContent = `${n.toLocaleString()} shortest solution${n > 1 ? 's' : ''} · none shorter`
        + (lines.length < n ? ` · ${lines.length.toLocaleString()} listed` : '')
        + (res.known ? '' : ` · ${fmtTime(secs)}`);
    }
    $('copyAllBtn').hidden = $('downloadBtn').hidden = lines.length === 0;
    $('solutionList').replaceChildren();
    showMore();
    markStale();
    if (lines.length) {
      selectSolution(0);
      scrollTo({ top: 0, behavior: 'smooth' });
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
    fontText($('solLabel'), `Solution ${(i + 1).toLocaleString()} of ${result.lines.length.toLocaleString()}`, { size: infoSize() });
    $('prevSol').disabled = $('nextSol').disabled = result.lines.length < 2;
    state.mode = 'replay';
    state.side = 'start';
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
    const u = b.offsetWidth / 500;                 // the board's clip box is 500 game pixels
    const pitch = 80 * u;
    const ox = b.offsetLeft + 10 * u + pitch / 2, oy = b.offsetTop + 10 * u + pitch / 2;
    const r = mv.col ? along : cross, c = mv.col ? cross : along;
    return { x: ox + c * pitch, y: oy + r * pitch };
  }

  function placeCursor(p, down = false) {
    guide.at = p;
    const el = $('guideCursor');
    const k = (el.offsetWidth || 52) / 52;
    el.style.transform = `translate(${p.x - HOT.x * k}px, ${p.y - HOT.y * k}px)`;
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
    document.querySelectorAll('[data-label]:not([data-mode])').forEach((el) =>
      fontText(el, el.dataset.label, 'gold' in el.dataset ? { ...GOLD_TEXT, size: 17 } : LABEL_TEXT));
    setProgress('', 0);
    $('depthDown').addEventListener('click', () => setDepth(state.maxDepth - 1));
    $('depthUpBtn').addEventListener('click', () => setDepth(state.maxDepth + 1));
    $('threads').appendChild(new Option(`Auto (${hardwareThreads})`, 0));
    for (const t of [1, 2, 4, 8, 12, 16, 24, 32]) {
      if (t < hardwareThreads) $('threads').appendChild(new Option(String(t), t));
    }

    fontText($('brandMain'), 'Mind Bender', { size: 21 });
    fontText($('brandSub'), 'Solver', { ...GOLD_TEXT, size: 21 });
    setChuzzle($('logo'), 4);
    addFace($('logo'), 7, 'normal', 'logo');
    setInterval(eyeTick, 100);
    requestAnimationFrame(frameLoop);
    let lastMove = null;
    addEventListener('pointermove', (e) => {
      if (!lastMove) requestAnimationFrame(() => { trackMouse(lastMove); lastMove = null; });
      lastMove = e;
    }, { passive: true });
    matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener('change', redrawFonts);
    attachEditor($('startBoard'), 'start');
    attachEditor($('goalBoard'), 'goal');

    $('tabStart').addEventListener('click', () => setSide('start'));
    $('tabGoal').addEventListener('click', () => setSide('goal'));
    $('unfatBtn').addEventListener('click', removeFat);
    $('undoBtn').addEventListener('click', undo);
    $('fillBtn').addEventListener('click', () => {
      pushUndo();
      state.goal.cells.fill(state.color);
      state.side = 'goal';
      syncStart();
      changed();
    });
    $('customBtn').addEventListener('click', enterCustom);
    $('coachNext').addEventListener('click', () => coachStep(state.coach + 1));
    $('coachSkip').addEventListener('click', endCoach);
    $('tourBtn').addEventListener('click', () => {
      $('helpDialog').close();
      enterCustom();
      coachStep(1);
    });
    $('scrambleBtn').addEventListener('click', () => {
      pushUndo();
      state.start = scrambled(state.start);
      state.follows = false;
      state.picked = null;
      state.side = 'start';
      changed();
    });
    $('resetBtn').addEventListener('click', () => {
      pushUndo();
      state.follows = true;
      state.picked = null;
      syncStart();
      state.side = 'start';
      changed();
    });
    $('shareBtn').addEventListener('click', async () => {
      writeHash();
      try {
        await navigator.clipboard.writeText(location.href);
        flash($('shareOrb'));
      } catch {
        prompt('Copy this link:', location.href);
      }
    });
    fontText($('helpTitleText'), 'How it works', { ...GOLD_TEXT, size: 26 });
    $('helpBtn').addEventListener('click', () => $('helpDialog').showModal());
    fontText($('pzTitleText'), 'Game puzzles', { ...GOLD_TEXT, size: 26 });
    $('puzzlesBtn').addEventListener('click', () => $('puzzlesDialog').showModal());
    for (const d of [$('helpDialog'), $('puzzlesDialog')]) {
      d.addEventListener('click', (e) => { if (e.target === d) d.close(); });
    }

    $('solveBtn').addEventListener('click', solve);
    $('cancelBtn').addEventListener('click', cancel);
    $('moreBtn').addEventListener('click', showMore);
    $('solutionList').addEventListener('click', (e) => {
      const li = e.target.closest('li');
      if (!li) return;
      selectSolution(+li.dataset.i);
      scrollTo({ top: 0, behavior: 'smooth' });
    });
    $('editBtn').addEventListener('click', () => { leaveReplay(); render(); });
    $('listBtn').addEventListener('click', () => $('results').scrollIntoView({ behavior: 'smooth', block: 'start' }));
    $('upBtn').addEventListener('click', () => scrollTo({ top: 0, behavior: 'smooth' }));
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
    new ResizeObserver(() => layoutStage()).observe($('stage'));
    addEventListener('resize', renderInfo);

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
      else if (e.key === '9') selectFatTool();
    });

    const linked = readHash();
    layoutStage();
    changed({ keepPreset: true });
    let seen = false;
    try { seen = localStorage.getItem('mb-tutorial') === 'done'; } catch { /* storage blocked */ }
    if (!seen && !linked) coachStep(1);
    warmUp();
  }

  init();
})();
