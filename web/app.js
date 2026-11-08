// web/app.js - board editor, the solver workers, results and the guided replay.
'use strict';

(() => {
  const COLORS = ['Red', 'Green', 'Blue', 'Orange', 'Yellow', 'Purple', 'White', 'Cyan'];
  const MAX_DEPTH = 12;
  const LIST_PAGE = 300;
  const $ = (id) => document.getElementById(id);
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

  // The Settings dialog's choices, kept in this browser. first: stop at the first shortest
  // solution (the default; every shorter length is still searched in full) or list them all.
  // threads: 0 = auto.
  const settings = { first: true, threads: 0 };
  try { Object.assign(settings, JSON.parse(localStorage.getItem('mb-settings') || '{}')); } catch { /* storage blocked */ }
  const saveSettings = () => {
    try { localStorage.setItem('mb-settings', JSON.stringify(settings)); } catch { /* storage blocked */ }
  };

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
    gear: '<path d="M12.2 2h-.4a2 2 0 0 0-2 2v.2a2 2 0 0 1-1 1.7l-.4.3a2 2 0 0 1-2 0l-.2-.1a2 2 0 0 0-2.7.7l-.2.4a2 2 0 0 0 .7 2.7l.2.1a2 2 0 0 1 1 1.7v.5a2 2 0 0 1-1 1.8l-.2.1a2 2 0 0 0-.7 2.7l.2.4a2 2 0 0 0 2.7.7l.2-.1a2 2 0 0 1 2 0l.4.3a2 2 0 0 1 1 1.7v.2a2 2 0 0 0 2 2h.4a2 2 0 0 0 2-2v-.2a2 2 0 0 1 1-1.7l.4-.3a2 2 0 0 1 2 0l.2.1a2 2 0 0 0 2.7-.7l.2-.4a2 2 0 0 0-.7-2.7l-.2-.1a2 2 0 0 1-1-1.8v-.5a2 2 0 0 1 1-1.7l.2-.1a2 2 0 0 0 .7-2.7l-.2-.4a2 2 0 0 0-2.7-.7l-.2.1a2 2 0 0 1-2 0l-.4-.3a2 2 0 0 1-1-1.7V4a2 2 0 0 0-2-2Z"/><circle cx="12" cy="12" r="3"/>',
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
  // chuzzle itself, and goes the way that keeps it on the board: the 2x2 never crosses an
  // edge, so of right n and left 6 - n exactly one fits (right 4 from column 1, not left 2).
  function dragOf(mv, board) {
    const f = mv.lanes.length === 2 ? board.fat : null;
    const at = f ? (mv.col ? f.y : f.x) : 0;
    const d = f ? (at + mv.n <= 4 ? mv.n : mv.n - 6) : (mv.n <= 3 ? mv.n : mv.n - 6);
    const lo = d > 0 ? 0 : -d, hi = d > 0 ? 5 - d : 5;
    let s = Math.max(lo, Math.min(hi, Math.round((5 - d) / 2)));
    let cross = mv.lanes[0];
    if (mv.lanes.length === 2) {
      cross += 0.5;
      if (f) s = at + 0.5;
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

  // The colour of the logo's or a palette chuzzle (drawn on its canvas, attachActor).
  function setChuzzle(el, color) {
    el._k = color;
    if (el._actor) el._actor.node._k = color;
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
  let tracked = null;           // the board cell the mouse is over (its eyes follow the mouse)
  let trackedKey = null;
  let lastPointer = -1e9;       // when the mouse last moved: the eyes then follow at full rate
  let starsAlive = false;       // a board drew the match outline's stars this frame: no idling
  let actorEls = null;          // the logo's and the palette's chuzzles, found again after a redraw
  function actors() {
    if (!actorEls || !actorEls.every((el) => el.isConnected)) actorEls = [...document.querySelectorAll('.chz')];
    return actorEls;
  }
  // The logo's and the palette's chuzzles, like the board's: while the mouse is inside the
  // square a chuzzle fills, its pupils turn to the mouse (from the square's centre); leaving
  // holds that direction until the next idle look.
  function followMouse(e) {
    const now = performance.now();
    for (const el of actors()) {
      const r = el.getBoundingClientRect(), s = eyeState(el._actor.key);
      const inside = e.clientX >= r.left && e.clientX < r.right && e.clientY >= r.top && e.clientY < r.bottom;
      if (!inside) {
        if (s.track) {
          s.track = false;
          s.lookAt = now + 5000 + rand(9000);
        }
        continue;
      }
      const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
      const len = Math.hypot(dx, dy);
      [s.dx, s.dy] = len < 0.5 ? [0, 0] : [dx / len, dy / len];
      s.lookAt = Infinity;                 // no idle look while the mouse is on it
      s.track = true;
    }
  }
  // The mouse over a board's chuzzle: its pupils turn to the mouse (from the centre of its
  // cell, or of the 2x2's four); leaving holds that direction until the next idle look.
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

  // ---------------------------------------------------------------- the chuzzle sprites
  // Every chuzzle on the page is drawn from the game's sprite sheets onto a canvas: each
  // board's on one canvas, the logo's and the palette's each on its own small one, the held
  // one and the particles on a layer over the page. A chuzzle is redrawn only when its pose
  // changes (a body frame every ~87 ms, a blink, the pupils moving), and while nothing but the
  // idle sway is going on, at most 12 times a second: the page then costs the browser a few
  // small blits a second, not a repaint of everything at the screen's rate.
  const SHEETS = {};
  for (const [name, file] of [['body', 'chuzzles'], ['huge', 'huge'], ['eyes', 'eyes'], ['pupils', 'pupils'],
    ['fizzy', 'fizzy'], ['arrows', 'arrows'], ['naked', 'naked'], ['wince', 'wince'], ['beauty', 'beauty'],
    ['cheeks', 'cheeks'], ['rosy', 'rosy'], ['fuzz', 'fuzz'], ['star', 'star'], ['splode', 'splode'],
    ['shockwave', 'shockwave']]) {
    const img = new Image();
    img.src = `assets/${file}.webp`;
    img.decode().then(() => { SHEETS[name] = img; }, () => {});
  }
  // Board::InitColors: the hilite tints (mColors2; the match outline's colour) and the 2x2's
  // hilite, also the cheeks' (mColors3); and the page's own body tints (gen_sprites.py), for
  // the shut lids and the fuzz
  const HILITE = [[1, 0.4, 0.4], [0.45, 0.9, 0.45], [0.55, 0.55, 1], [1, 0.85, 0.3], [1, 1, 0.55], [1, 0.65, 1], [1, 1, 1], [0.5, 1, 1]];
  const HILITE3 = [[1, 0.45, 0.45], [0.25, 0.9, 0.5], [0.55, 0.55, 1], [1, 0.85, 0.55], [1, 1, 0.75], [1, 0.6, 1], [1, 1, 1], [0.65, 1, 1]];
  const BODY = [[0.7, 0.03, 0.03], [0.03, 0.68, 0.03], [0.03, 0.03, 1], [1, 0.41, 0.01], [1, 1, 0], [0.81, 0.03, 0.81], [1, 1, 1], [0.34, 0.77, 0.87]];

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
    const track = s.track || trackedKey === key;
    if (!track && now >= s.lookAt) {
      s.lookAt = now + 5000 + rand(9000);
      [s.dx, s.dy] = randomLook();
    }
    // the idle look eases in over ~0.5 s, the mouse is followed at once
    const k = 1 - Math.exp(-dt / (track ? 25 : 120));
    s.px = (s.px ?? s.dx) + (s.dx - (s.px ?? s.dx)) * k;
    s.py = (s.py ?? s.dy) + (s.dy - (s.py ?? s.dy)) * k;
  }

  // A chuzzle's pose this frame: body frame, face bob, lids, pupils (to a quarter of a game
  // pixel); `sig` changes when drawing it again would show anything new, `busy` while an
  // antic animates it every frame.
  function pose(node, now, dt, held) {
    const piece = node._piece, f = look(node._key), st = eyeState(node._key), a = antics.get(node._key);
    updateEyes(st, node._key, now, dt);
    let frame, bob;
    if (node._f != null) [frame, bob] = [node._f, bounce(f, piece, now)[1]];
    else if (held) {
      if (node._hold == null) node._hold = bounce(f, piece, now);
      [frame, bob] = node._hold;
    } else {
      node._hold = null;
      [frame, bob] = bounce(f, piece, now);
    }
    // poked: squashed, the frames a move down would show (Chuzzle::Draw with mMoveDown)
    if (a && a.moveDown > 0) frame = piece ? 10 + Math.floor(a.moveDown / 2) : 20 + Math.floor(a.moveDown);
    const shut = now < (st.blinkUntil || 0) || !!(a && a.shut);
    const [tx, ty] = EYE_PUPIL_TRAVEL[f.es];
    const px = Math.round(st.px * tx * 4), py = Math.round(st.py * ty * 4);
    const busy = !!(a && (a.performing || a.mouth > 0 || a.naked));
    return { frame, bob, shut, busy, sig: ((((frame * 4 + bob) * 2 + (shut ? 1 : 0)) * 64 + px + 32) * 64 + py + 32) * 8 + node._k };
  }

  // A piece of a sheet at the size it has on the screen, cut once and kept: every chuzzle is
  // then a few 1:1 blits of small images. Drawing scaled from the sheets themselves on every
  // frame was most of the page's drawing cost (in Chrome a scaled draw from a sheet is 9x a
  // blit of an ImageBitmap, in Firefox 10x). The cut is a canvas until its ImageBitmap is
  // made (Chrome blits a bitmap 4x faster than a canvas). rgb tints it; disc = [cx, cy, r] in
  // the sheet's pixels keeps only that circle of it.
  // The iris and its ring in each 14 px pupil frame: [cx, cy, r], from the frame's opaque box
  // (the art is not centred the same in every frame: the first sits a pixel further left).
  let pupilDiscs = null;
  function pupilDisc(frame) {
    if (!pupilDiscs) {
      const sheet = SHEETS.pupils, c = document.createElement('canvas');
      c.width = sheet.width;
      c.height = sheet.height;
      const g = c.getContext('2d');
      g.drawImage(sheet, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data;
      pupilDiscs = [];
      for (let f = 0; f * 14 < c.width; f++) {
        let x0 = 14, y0 = 14, x1 = -1, y1 = -1;
        for (let y = 0; y < 14; y++) {
          for (let x = 0; x < 14; x++) {
            if (d[(y * c.width + f * 14 + x) * 4 + 3] < 128) continue;
            x0 = Math.min(x0, x);
            y0 = Math.min(y0, y);
            x1 = Math.max(x1, x);
            y1 = Math.max(y1, y);
          }
        }
        pupilDiscs.push(x1 < 0 ? [7, 7, 7] : [(x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2, Math.max(x1 - x0, y1 - y0) / 2]);
      }
    }
    return pupilDiscs[frame] || pupilDiscs[0];
  }
  const cuts = new Map();
  function cut(name, sx, sy, sw, sh, dw, dh, rgb, disc) {
    const key = `${name}|${sx}|${sy}|${dw}|${dh}|${rgb || ''}|${disc || ''}`;
    let c = cuts.get(key);
    if (!c) {
      const img = rgb ? tinted(name, rgb) : SHEETS[name];
      if (!img) return null;
      if (cuts.size > 1500) cuts.clear();
      c = document.createElement('canvas');
      c.width = dw;
      c.height = dh;
      const g = c.getContext('2d');
      g.imageSmoothingQuality = 'high';
      g.drawImage(img, sx, sy, sw, sh, 0, 0, dw, dh);
      if (disc) {
        g.globalCompositeOperation = 'destination-in';
        g.beginPath();
        g.arc(disc[0] * dw / sw, disc[1] * dh / sh, disc[2] * dw / sw, 0, 2 * Math.PI);
        g.fill();
      }
      cuts.set(key, c);
      if (self.createImageBitmap) createImageBitmap(c).then((b) => { if (cuts.get(key) === c) cuts.set(key, b); }, () => {});
    }
    return c;
  }
  const blit = (ctx, c, x, y) => { if (c) ctx.drawImage(c, x, y); };

  // Draws one chuzzle (Chuzzle::Draw and DrawEyes): the body frame, the face with its lids or
  // pupils, and whatever antic it is in. x, y is its cell's top-left on the canvas.
  function drawChuzzle(ctx, node, p, now, dt, s, R, cellPx, padPx, lifted) {
    const piece = node._piece, f = look(node._key), st = eyeState(node._key), a = antics.get(node._key), k = node._k;
    const dpr = s / layout.u;
    let x = node._x0 != null ? node._x0 * s : padPx + node._c * cellPx;
    let y = node._y0 != null ? node._y0 * s : padPx + node._r * cellPx;
    // a lane slide (replay) or a drag (in CSS pixels, so times dpr)
    if (node._off) {
      if (node._col) y += node._off * dpr;
      else x += node._off * dpr;
    }
    // easing (a chuzzle settling into its cell after a drop or swap): the offset _ox/_oy
    // moves to _tx/_ty, the size _sc back to 1
    if (node._ox || node._oy || node._tx || node._ty) {
      const e = 1 - Math.exp(-dt / 70), tx = node._tx || 0, ty = node._ty || 0;
      node._ox = Math.abs(tx - (node._ox || 0)) < 0.2 ? tx : (node._ox || 0) + (tx - (node._ox || 0)) * e;
      node._oy = Math.abs(ty - (node._oy || 0)) < 0.2 ? ty : (node._oy || 0) + (ty - (node._oy || 0)) * e;
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
    if (a && a.mouth > 0) {          // its mouth open: it shakes (mMouthOpen)
      x += rnd(-3, 3) * a.mouth * s;
      y += rnd(-3, 3) * a.mouth * s;
    }
    const half = (piece ? 80 : 40) * s;
    const bx = piece ? R(x - 9.5 * s) : R(x), by = piece ? R(y - 8.5 * s) : R(y), bw = piece ? R(179 * s) : R(cellPx);
    if (sc > 1) {
      ctx.save();
      ctx.translate(x + half, y + half);
      ctx.scale(sc, sc);
      ctx.translate(-x - half, -y - half);
      const sh = shadowOf(piece, p.frame, k, s, dpr);
      ctx.globalAlpha = Math.min(1, (sc - 1) / 0.12);
      ctx.drawImage(sh, bx - sh._m, by - sh._m);
      ctx.globalAlpha = 1;
    }
    // the body; bald after the huge sneeze, the fuzz fading back in over it
    const fuzz = a && a.naked ? Math.max(0, 1 - a.fuzzGone) : 1;
    if (a && a.naked) blit(ctx, cut('naked', 0, 0, 64, 64, R(64 * s), R(64 * s)), R(x + 8 * s), R(y + 15 * s));
    if (fuzz > 0) {
      if (fuzz < 1) ctx.globalAlpha = fuzz;
      if (piece) blit(ctx, cut('huge', p.frame * 180, k * 180, 179, 179, bw, bw), bx, by);
      else blit(ctx, cut('body', p.frame * 80, k * 80, 80, 80, bw, bw), bx, by);
      ctx.globalAlpha = 1;
    }
    // the face (DrawEyes): `at` is the cell's top-left one pixel down; the 2x2's face sits at
    // (44, 71) of its block. The face's centre is 40 right and 40 down of it, plus the bob.
    const ax = piece ? x + 44 * s : x, ay = piece ? y + 71 * s : y + s;
    const bob = p.bob * s, cx = ax + 40 * s, cy = ay + 40 * s + f.oy * s + bob;
    const show = a && a.performing;
    if (a && a.beauty) {
      blit(ctx, cut('beauty', 0, 0, 114, 121, R(114 * s), R(121 * s)), R(cx - 10 * s - 57 * s), R(cy - 15 * s - 60.5 * s));
    } else if (a && a.wince) {
      blit(ctx, cut('wince', 0, 0, 40, 12, R(40 * s), R(12 * s)), R(cx + a.ox * s - 20 * s), R(cy + 20 * s + a.oy * s - 6 * s));
    } else {
      const aox = show ? a.ox * s : 0, aoy = show ? a.oy * s : 0, up = a && a.sneezing ? 10 * s : 0;
      const ex = R(cx - 42.5 * s + aox), ey = R(cy - 44 * s + aoy - up), ew = R(85 * s), eh = R(88 * s);
      if (p.shut) {
        blit(ctx, cut('eyes', f.es * 85, 0, 85, 88, ew, eh, BODY[k]), ex, ey);   // the lids, in its own colour
      } else {
        blit(ctx, cut('eyes', f.es * 85, 0, 85, 88, ew, eh), ex, ey);
        // the pupil texture is a rounded square (an 11 px iris in a dark ring, cut at the corners
        // by its alpha), which reads as round at the game's 1:1 but not scaled up: keep its disc
        const [tx, ty] = EYE_PUPIL_TRAVEL[f.es], ps = R(14 * s), pupil = cut('pupils', f.ep * 14, 0, 14, 14, ps, ps, null, pupilDisc(f.ep));
        EYE_PUPIL_CENTERS[f.es].forEach(([pcx, pcy], i) => {
          // sneezing: cross-eyed (each pupil turned inwards)
          const lx = a && a.sneezing ? (i ? -tx : tx) : st.px * tx, ly = a && a.sneezing ? 0 : st.py * ty;
          blit(ctx, pupil, R(ex + (pcx - 7 + lx) * s), R(ey + (pcy - 7 + ly) * s));
        });
      }
    }
    if (a && a.cheeks && SHEETS.cheeks) {               // the hurk: puffed cheeks
      const cs = Math.max(0, Math.min(1, Math.abs(a.scale)));
      if (cs > 0.02) {
        const w = R(98 * s * cs), h = R(50 * s * cs), ccx = ax + 40 * s, ccy = ay + 65 * s + bob;
        const tint = tinted('cheeks', HILITE3[k]);
        if (tint) ctx.drawImage(tint, R(ccx - w / 2), R(ccy - h / 2), w, h);
        if (SHEETS.rosy) ctx.drawImage(SHEETS.rosy, R(ccx - w / 2), R(ccy - h / 2), w, h);
      }
    }
    if (sc > 1) ctx.restore();
  }

  // The shadow under a chuzzle (lifted or settling on a board: blur 8, 6 down; the logo's and
  // the palette's at rest: blur 3, 2 down, CSS pixels), baked once per body frame (a canvas
  // shadow blur on every frame of a drag is what made the drag heavy). dpr: device pixels per
  // CSS pixel.
  const shadows = new Map();
  function shadowOf(piece, frame, k, s, dpr, blur = 8, dy = 6, alpha = 0.55) {
    const key = `${piece ? 'p' : 'n'}${frame}|${k}|${s.toFixed(3)}|${blur}`;
    let c = shadows.get(key);
    if (!c) {
      if (shadows.size > 400) shadows.clear();
      const size = Math.round((piece ? 179 : 80) * s), m = Math.ceil((2 * blur + dy) * dpr);
      c = document.createElement('canvas');
      c.width = size + 2 * m;
      c.height = size + 2 * m;
      c._m = m;
      const g = c.getContext('2d');
      g.shadowColor = `rgba(0, 0, 0, ${alpha})`;
      g.shadowBlur = blur * dpr;
      g.shadowOffsetY = dy * dpr;
      g.shadowOffsetX = 4 * size;        // the body itself lands off the canvas: only its shadow
      if (piece) g.drawImage(SHEETS.huge, frame * 180, k * 180, 179, 179, m - 4 * size, m, size, size);
      else g.drawImage(SHEETS.body, frame * 80, k * 80, 80, 80, m - 4 * size, m, size, size);
      shadows.set(key, c);
    }
    return c;
  }

  // A sprite tinted the way the game's SetColor does (multiplied), cached per colour.
  const tints = new Map();
  function tinted(name, rgb) {
    const key = `${name}|${rgb}`;
    let c = tints.get(key);
    if (!c && SHEETS[name]) {
      const img = SHEETS[name];
      c = document.createElement('canvas');
      c.width = img.width;
      c.height = img.height;
      const k = c.getContext('2d');
      k.drawImage(img, 0, 0);
      k.globalCompositeOperation = 'multiply';
      k.fillStyle = `rgb(${rgb.map((v) => Math.round(v * 255)).join(',')})`;
      k.fillRect(0, 0, c.width, c.height);
      k.globalCompositeOperation = 'destination-in';
      k.drawImage(img, 0, 0);
      tints.set(key, c);
    }
    return c;
  }

  // The game's match outline (Board::SparkleGroups), the stars it runs round the chuzzles a
  // drag would match. Its points: a ring 45 game pixels from each marked chuzzle's centre, one
  // every 20 degrees (90 and every 10 round a 2x2 chuzzle), less those inside another marked
  // chuzzle (44; 89 for a 2x2), crowded by more than three (within 85) or on top of another
  // point (5); so round neighbours the outline runs in and out between them. A 2x2 footprint
  // over single chuzzles (the 2x2 tool's spot, a 2x2's drop target) is its four cells, so
  // the chain goes round each and dips in between them. Each mark is a chain of its own,
  // and every chain runs at once: clockwise round each chuzzle, the four of a footprint in
  // clockwise order (top left, top right, bottom right, bottom left), each ring starting at
  // its first point past the direction of the next ring, so it ends beside where the next
  // one starts and the last ring beside the first: a loop the stars can run round for ever.
  // Two marks side by side (a 2x2 lifted next to where it would land) start their loops at
  // the same corner, so they run in step. A lone chuzzle's ring starts top left.
  // marks: {r, c, big, fat, k} in cells, k the colour the spot will have; fat: the 2x2
  // chuzzle itself. The points are in board game pixels.
  // a mark's identity is its place (its chain keeps running when only its colours change)
  const markId = (m) => `${m.r},${m.c},${m.big ? 1 : 0}${m.fat ? 'f' : ''}${m.cells ? ':' + m.cells.map((q) => `${q[0]}${q[1]}`).join('') : ''}`;
  const markKey = (m) => `${markId(m)},${m.cells ? m.cells.map((q) => q[2]).join('') : m.k}`;
  const sparkleKey = (marks) => marks.map(markKey).join(';');
  function sparkleChains(marks) {
    const dist = (p, q) => Math.hypot(p.x - q.x, p.y - q.y);
    const groups = marks.map((m) => {
      const cs = m.fat ? [{ x: m.c * 80 + 80, y: m.r * 80 + 80, big: true, k: m.k }]
        : (m.cells || (m.big ? [[0, 0], [0, 1], [1, 1], [1, 0]] : [[0, 0]])).map(([dr, dc, k]) => ({ x: (m.c + dc) * 80 + 40, y: (m.r + dr) * 80 + 40, big: false, k: k ?? m.k }));
      return { key: markId(m), cs, pts: [] };
    });
    const all = groups.flatMap((g) => g.cs), kept = [];
    for (const g of groups) {
      g.cs.forEach((c, i) => {
        const rad = c.big ? 90 : 45, step = c.big ? 10 : 20;
        const nx = g.cs[(i + 1) % g.cs.length];
        const to = g.cs.length > 1 ? Math.atan2(nx.y - c.y, nx.x - c.x) * 180 / Math.PI : 225;
        const a0 = Math.ceil(to / step) * step;     // angles clockwise (y down on the canvas)
        for (let j = 0; j < 360 / step; j++) {
          const t = (a0 + j * step) * Math.PI / 180;
          const p = { x: c.x + Math.cos(t) * rad, y: c.y + Math.sin(t) * rad, k: c.k };
          let crowd = 0, inside = false;
          for (const o of all) {
            const d = dist(p, o);
            if (o.big) {
              if (d < 89) inside = true;
            } else {
              if (d < 44) inside = true;
              if (d < 85) crowd++;
            }
          }
          if (inside || crowd > 3 || kept.some((q) => dist(p, q) < 5)) continue;
          kept.push(p);
          g.pts.push(p);
        }
      });
      // the box the stars can reach (a star is 21 pixels across, turned, up to 2 off its point)
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const p of g.pts) {
        x0 = Math.min(x0, p.x - 18);
        y0 = Math.min(y0, p.y - 18);
        x1 = Math.max(x1, p.x + 20);
        y1 = Math.max(y1, p.y + 20);
      }
      g.cbox = [x0, y0, x1, y1];
    }
    return groups.filter((g) => g.pts.length);
  }

  // The stars on a board: a runner per chain, 100 updates a second whatever the frame rate,
  // 2.5 updates a point (the game: one; slower here, the outline being on for as long as a
  // mark stands, not one burst), lighting a fizzy star in the chuzzle's hilite colour (half
  // of them up to a quarter darker) at each point it passes: 0.75-1 in scale, turned up to
  // 45 degrees, up to 2 pixels off its point, the first of a chain at alpha 0.5 and 0.1
  // brighter each up to 1 (the game's ramp), fading 0.015 an update (the game: 0.02), so a
  // trail of some 27 points follows the runner round and round. A mark that goes stops its
  // runner and its trail fades out. Returns the chains with stars alive.
  const STAR_STEP = 2.5;
  function stepSparkles(el, marks, now, dt) {
    if (!SHEETS.fizzy) return [];
    const key = sparkleKey(marks);
    let sp = el._spark;
    if (!sp) {
      if (!marks.length) return [];
      sp = el._spark = { key: null, groups: [], acc: 0, ticks: 0, drawn: 0 };
    }
    if (sp.key !== key) {
      sp.key = key;
      const old = new Map(sp.groups.map((g) => [g.key, g]));
      const groups = [];
      for (const ch of sparkleChains(marks)) {
        let g = old.get(ch.key);
        if (g) old.delete(ch.key);
        else g = { key: ch.key, pos: 0, spawned: 0, stars: [] };
        g.live = true;
        g.pts = ch.pts;        // new points, new colours; its stars and runner go on
        g.cbox = ch.cbox;
        groups.push(g);
      }
      for (const g of old.values()) {
        g.live = false;
        if (g.stars.length) groups.push(g);
      }
      sp.groups = groups;
    }
    sp.acc += Math.min(dt, 100);
    for (; sp.acc >= 10; sp.acc -= 10) {
      sp.ticks++;
      for (const g of sp.groups) {
        for (const st of g.stars) st.a -= 0.015;
        if (!g.live) continue;
        g.pos += 1 / STAR_STEP;
        for (; g.spawned < g.pos; g.spawned++) {
          const p = g.pts[g.spawned % g.pts.length];
          const dark = Math.random() < 0.5 ? 1 - 0.25 * Math.floor(Math.random() * 4) / 3 : 1;   // four shades, so few tinted stars
          g.stars.push({ x: p.x + Math.random() * 2, y: p.y + Math.random() * 2, sc: 0.75 + Math.random() * 0.25,
            rot: (Math.random() * 90 - 45) * Math.PI / 180, a: Math.min(1, 0.5 + 0.1 * g.spawned),
            rgb: (HILITE[p.k] ?? HILITE[6]).map((v) => Math.round(v * dark * 100) / 100) });
        }
      }
    }
    for (const g of sp.groups) {
      g.stars = g.stars.filter((st) => st.a > 0);
      // the box to clear: the chain's, and whatever stars of an older chain are still fading
      const b = g.cbox.slice();
      for (const st of g.stars) {
        b[0] = Math.min(b[0], st.x - 18);
        b[1] = Math.min(b[1], st.y - 18);
        b[2] = Math.max(b[2], st.x + 20);
        b[3] = Math.max(b[3], st.y + 20);
      }
      g.box = b;
    }
    sp.groups = sp.groups.filter((g) => g.live || g.stars.length);
    if (!sp.groups.length) el._spark = null;
    return sp.groups.filter((g) => g.stars.length);
  }

  function drawStars(ctx, groups, s, padPx) {
    for (const g of groups) {
      for (const st of g.stars) {
        const sc = st.sc * s;
        ctx.globalAlpha = st.a;
        ctx.setTransform(sc, 0, 0, sc, padPx + st.x * s, padPx + st.y * s);
        ctx.rotate(st.rot);
        ctx.drawImage(tinted('fizzy', st.rgb), -10.5, -10);
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
  }

  // The ring round the palette item in use, after the game's hint circle (Animator_Hint: a
  // circle with a black copy under it, down and right) but thin, close round the fur (the
  // body reaches 38 game pixels from its centre, the 2x2's 84) and in the chuzzle's own colour.
  // (cx, cy) in canvas pixels.
  function drawHint(ctx, cx, cy, big, k, s) {
    const rad = (big ? 90 : 44) * s, w = (big ? 5 : 4) * s;
    ctx.lineWidth = w;
    for (const [style, o] of [['rgba(0, 0, 0, 0.6)', 2 * s], [`rgb(${BODY[k].map((v) => Math.round(v * 255)).join(',')})`, 0]]) {
      ctx.strokeStyle = style;
      ctx.beginPath();
      ctx.arc(cx + o, cy + o, rad, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }

  // A faint chuzzle of the colour a drop's emptied spot will get.
  function drawGhost(ctx, r, c, k, big, s, R, cellPx, padPx) {
    ctx.globalAlpha = 0.45;
    const x = padPx + c * cellPx, y = padPx + r * cellPx;
    if (big) blit(ctx, cut('huge', 0, k * 180, 179, 179, R(179 * s), R(179 * s)), R(x - 9.5 * s), R(y - 8.5 * s));
    else blit(ctx, cut('body', 0, k * 80, 80, 80, R(cellPx), R(cellPx)), R(x), R(y));
    ctx.globalAlpha = 1;
  }

  // What a click on the goal would paint, under the pointer: the cell (or the whole 2x2 it is
  // part of), or with the 2x2 tool the block the piece would cover; k the colour it gets.
  function paintPreview() {
    const h = $('goalBoard')._hover;
    if (!h || state.mode !== 'edit' || state.preset) return null;
    const b = state.goal;
    if (state.tool === 'piece') {
      const r = Math.min(h.r, 4), c = Math.min(h.c, 4), fat = !!b.fat && b.fat.y === r && b.fat.x === c;
      return { r, c, big: true, fat, k: pieceColour(b) ?? state.color };
    }
    if (inPiece(b.fat, h.r, h.c)) return { r: b.fat.y, c: b.fat.x, big: true, fat: true, k: state.color };
    return { r: h.r, c: h.c, big: false, k: state.color };
  }

  // One board's chuzzles. Only what changed is cleared and drawn again: the cells whose pose
  // changed (the 2x2's sprite reaches into the cells round it, so it and they go together), a
  // sliding lane's strip (a replay move, with the cells its edges touch); the whole canvas
  // when a chuzzle settles after a drop, when the held one is lifted or put back, and while
  // the match outline's stars are alive or its marks change. A drag, with the pointer moving
  // and nothing else, redraws nothing here. Idle, a redraw waits for the 80 ms tick.
  function drawBoard(el, now, dt, maySkip) {
    const cv = boardCanvas(el), ctx = el._ctx;
    const dpr = window.devicePixelRatio || 1, s = layout.u * dpr;
    const size = Math.round(500 * s);
    let full = false;
    if (cv.width !== size) {
      cv.width = size;
      cv.height = size;
      full = true;
    }
    if (!SHEETS.body || !SHEETS.huge || !SHEETS.eyes || !SHEETS.pupils) return;
    const R = Math.round, cellPx = 80 * s, padPx = 10 * s;
    // the match outline marks the picked chuzzle, where a drop lands, the spots it fills and
    // what a click on the goal will paint
    const marks = [];
    const held = dragLayer.node && dragLayer.node.parentNode === el ? dragLayer.node : null;
    if (el._target) marks.push({ r: el._target.r, c: el._target.c, big: el._target.piece, k: dragLayer.node ? dragLayer.node._k : 6 });
    if (el._fills) {
      if (held && held._piece) {   // the 2x2's emptied spot: one chain round the cells it leaves, each in the colour it gets
        const cells = [];
        for (const [dr, dc] of [[0, 0], [0, 1], [1, 1], [1, 0]]) {
          const f = el._fills.find((q) => q.r === held._r + dr && q.c === held._c + dc);
          if (f) cells.push([dr, dc, f.k]);
        }
        if (cells.length) marks.push({ r: held._r, c: held._c, big: true, cells, k: cells[0][2] });
      } else for (const f of el._fills) marks.push({ r: f.r, c: f.c, big: false, k: f.k });
    }
    const hover = el.id === 'goalBoard' ? paintPreview() : null;
    if (hover) marks.push(hover);
    if (held !== el._heldWas) full = true;
    el._heldWas = held;
    const nodes = [], dirty = [], moving = [];
    let piece = null;
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
      if (node === held) continue;           // drawn on its own canvas over the page
      nodes.push(node);
      if (node._piece) piece = node;
      if (node.classList.contains('picked')) marks.push({ r: node._r, c: node._c, big: node._piece, fat: node._piece, k: node._k });
      if (node._off) moving.push(node);
      if (node._ox || node._oy || node._tx || node._ty || node._sc > 1) full = true;
      node._pose = pose(node, now, dt, false);
      if (node._pose.busy || node._pose.sig !== node._sig) dirty.push(node);
    }
    if (el._movingWas && !moving.length) full = true;    // the slid sprites' last positions
    el._movingWas = moving.length > 0;
    // the outline's stars run on whatever the frame rate, drawn at most every second update
    // (50 a second: a composite per frame is the cost, and a star only fades between frames);
    // a mark set that changes (or whose last stars go) means a whole redraw, the stars' own
    // boxes are cleared and redrawn while they are alive
    const keyWas = el._spark ? el._spark.key : null;
    const stars = stepSparkles(el, marks, now, dt);
    if ((el._spark ? el._spark.key : null) !== keyWas) full = true;
    if (stars.length) starsAlive = true;
    const fresh = stars.length > 0 && el._spark.ticks - el._spark.drawn >= 2;
    if (!full && !moving.length && !fresh && (!dirty.length || maySkip)) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.imageSmoothingQuality = 'high';
    const rectOf = (n) => {
      const dx = n._off && !n._col ? n._off * dpr : 0, dy = n._off && n._col ? n._off * dpr : 0;
      return n._piece ? [padPx + n._c * cellPx - 9.5 * s + dx, padPx + n._r * cellPx - 8.5 * s + dy, 179 * s, 179 * s]
        : [padPx + n._c * cellPx + dx, padPx + n._r * cellPx + dy, cellPx, cellPx];
    };
    const hits = (a, b) => a[0] < b[0] + b[2] && a[0] + a[2] > b[0] && a[1] < b[1] + b[3] && a[1] + a[3] > b[1];
    let draw = nodes, set = null;
    const zones = [];
    if (full) ctx.clearRect(0, 0, size, size);
    else {
      set = new Set(dirty);
      if (moving.length) {
        const col = moving[0]._col;
        let lo = Infinity, hi = -Infinity;
        for (const n of moving) {
          const i = col ? n._c : n._r;
          lo = Math.min(lo, i);
          hi = Math.max(hi, i + (n._piece ? 1 : 0));
        }
        const a = padPx + lo * cellPx - 10 * s, b = padPx + (hi + 1) * cellPx + 10 * s;
        zones.push(col ? [a, 0, b - a, size] : [0, a, size, b - a]);
      }
      for (const g of stars) zones.push([padPx + g.box[0] * s, padPx + g.box[1] * s, (g.box[2] - g.box[0]) * s, (g.box[3] - g.box[1]) * s]);
      if (zones.length) for (const n of nodes) {
        const q = rectOf(n);
        if (zones.some((z) => hits(q, z))) set.add(n);
      }
      if (piece) {
        const near = (n) => n !== piece && Math.abs(n._r - piece._r - 0.5) <= 1.5 && Math.abs(n._c - piece._c - 0.5) <= 1.5;
        if (set.has(piece) || [...set].some(near)) {
          set.add(piece);
          for (const n of nodes) if (near(n)) set.add(n);
        }
      }
      for (const z of zones) ctx.clearRect(...z.map(R));
      for (const n of set) ctx.clearRect(...rectOf(n).map(R));
      draw = nodes.filter((n) => set.has(n));
    }
    if (el._fills) {
      for (const f of el._fills) {
        const q = [padPx + f.c * cellPx, padPx + f.r * cellPx, cellPx, cellPx];
        if (full || zones.some((z) => hits(q, z)) || [...set].some((n) => n._r === f.r && n._c === f.c)) drawGhost(ctx, f.r, f.c, f.k, false, s, R, cellPx, padPx);
      }
    }
    const top = [];
    for (const node of draw) {
      if (node._sc > 1) top.push(node);
      else drawChuzzle(ctx, node, node._pose, now, dt, s, R, cellPx, padPx, false);
      node._sig = node._pose.sig;
    }
    for (const node of top) drawChuzzle(ctx, node, node._pose, now, dt, s, R, cellPx, padPx, false);
    if (stars.length) {
      drawStars(ctx, stars, s, padPx);
      el._spark.drawn = el._spark.ticks;
    }
  }

  // The logo's and the palette's chuzzles, each on its own canvas inside its element (the
  // element's box is 80 game pixels across, the 2x2's 180 with the sprite from (-9.5, -8.5)).
  // The canvas reaches 10 game pixels past the box (style.css), room for the shadow and for
  // the ring round the palette item in use.
  const ACTOR_MARGIN = 10;
  function attachActor(el, kind, key) {
    const cv = document.createElement('canvas');
    cv.className = 'chz-canvas';
    cv.setAttribute('aria-hidden', 'true');
    el.appendChild(cv);
    el.classList.add('chz');
    const piece = kind !== 'normal';
    el._actor = {
      key, units: piece ? 180 : 80, margin: ACTOR_MARGIN, swatch: el.classList.contains('swatch'),
      cv, ctx: cv.getContext('2d'), w: 0, measured: 0, lit: -1,
      node: { _k: el._k ?? 0, _key: key, _piece: piece, _x0: piece ? 9.5 : 0, _y0: piece ? 8.5 : 0, _r: 0, _c: 0 },
    };
    actorEls = null;
    el.addEventListener('pointerdown', (e) => { if (e.button === 0) clickActor(el); });
    el.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      pokeActor(el);
    });
  }
  function drawActor(el, now, dt, maySkip) {
    const A = el._actor, node = A.node;
    if (!A.w || now - A.measured > 1000) {
      A.w = el.clientWidth;
      A.measured = now;
    }
    if (!A.w || !SHEETS.body || !SHEETS.huge || !SHEETS.eyes || !SHEETS.pupils) return;
    const dpr = window.devicePixelRatio || 1, s = A.w * dpr / A.units, m = A.margin * s;
    const size = Math.round((A.units + 2 * A.margin) * s);
    let full = false;
    if (A.cv.width !== size) {
      A.cv.width = size;
      A.cv.height = size;
      full = true;
    }
    const hint = A.swatch && el.getAttribute('aria-checked') === 'true', lit = hint ? node._k : -1;
    if (lit !== A.lit) full = true;
    A.lit = lit;
    const p = pose(node, now, dt, false);
    if (!full && !p.busy && (p.sig === node._sig || maySkip)) return;
    node._sig = p.sig;
    A.ctx.setTransform(1, 0, 0, 1, 0, 0);
    A.ctx.clearRect(0, 0, size, size);
    A.ctx.imageSmoothingQuality = 'high';
    A.ctx.setTransform(1, 0, 0, 1, m, m);
    const sh = shadowOf(node._piece, p.frame, node._k, s, dpr, 3, 2, 0.5);
    blit(A.ctx, sh, -sh._m, -sh._m);
    drawChuzzle(A.ctx, node, p, now, dt, s, Math.round, 80 * s, 0, false);
    // on the fur's own centre (the body sits 40, 40 in its frame, the 2x2's 89, 92)
    if (hint) drawHint(A.ctx, (node._piece ? 90.5 : 40.5) * s, (node._piece ? 92 : 40) * s, node._piece, node._k, s);
    A.ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // ---------------------------------------------------------------- antics (Chuzzle.cpp)
  // Clicking the logo's or a palette chuzzle again and again makes it perform, as the game's
  // do (Board::OnMouseDown): the 2nd and 3rd click a poke, the 4th the oggle, the 6th a cough,
  // the 8th a small sneeze, the 10th the huge one that blows its fuzz off; the 2x2: a squeeze,
  // a hurk (6th and 10th), the long hurk (8th), and on the 11th its beauty show. A right click
  // pokes it (a squash and a note). Clicking another one starts over. The game runs its antics
  // at 100 updates a second; PerformAntic is followed frame by frame.
  const antics = new Map();
  function antic(key) {
    let a = antics.get(key);
    if (!a) {
      a = { performing: false, kind: 0, frame: 0, ox: 0, oy: 0, scale: 0, wince: false, sneezing: false, shut: false,
        shutTimer: 0, mouth: 0, naked: false, fuzzGone: 0, cheeks: false, beauty: false, moveDown: 0, el: null, piece: false, k: 0 };
      antics.set(key, a);
    }
    return a;
  }
  let lastClicked = null, clicks = 0;
  function startAntic(a, kind, oy = 0) {
    a.performing = true;
    a.kind = kind;
    a.frame = 0;
    a.ox = 0;
    a.oy = oy;
  }
  function clickActor(el) {
    const a = antic(el._actor.key);
    a.el = el;
    a.piece = el._actor.node._piece;
    a.k = el._actor.node._k;
    if (lastClicked !== el) {
      lastClicked = el;
      clicks = 0;
    }
    if (a.performing) return;                 // busy: the click does not count
    const n = clicks;
    if (!a.piece) {
      if (n === 1 || n === 2) {               // AnticPoke
        a.shut = true;
        a.shutTimer = 30;
        a.mouth = 1;
        play('pokechuzzle');
      } else if (n === 3) {                   // AnticOggle
        startAntic(a, 0);
        a.mouth = 1;
        play('oggleoggle');
      } else if (n === 5) startAntic(a, 1);   // AnticCough
      else if (n === 7) {                     // AnticSmallSneeze
        startAntic(a, 2, 10);
        a.sneezing = true;
        play('smallsneeze');
      } else if (n === 9) {                   // AnticHugeSneeze
        startAntic(a, 3, 10);
        a.sneezing = true;
        play('hugesneeze');
      }
      clicks = n >= 9 ? 0 : n + 1;
    } else {
      if (n === 1 || n === 2) {               // AnticSqueeze
        startAntic(a, 4);
        a.mouth = 1;
        a.shut = true;
        a.shutTimer = 30;
        play('pokechuzzle', 0.75);
      } else if (n === 5 || n === 9 || n === 7) {   // AnticHurk, AnticHurkLong
        startAntic(a, n === 7 ? 6 : 5);
        a.mouth = 1;
        a.scale = 0;
        a.shut = true;
        a.shutTimer = 80;
        a.cheeks = true;
        play('hurk');
      } else if (n === 10) {                  // AnticBig
        startAntic(a, 7);
        a.mouth = 2.5;
        a.beauty = true;
        beautyBurst(a);
        play('sweetrelief');
      }
      clicks = n >= 10 ? 0 : n + 1;
    }
  }
  // a right click: the chuzzle shuts its eyes, squashes down and squeaks a note (the pitch
  // from where on the board it was; here from which chuzzle)
  function pokeActor(el) {
    const a = antic(el._actor.key);
    a.el = el;
    a.piece = el._actor.node._piece;
    a.k = el._actor.node._k;
    a.moveDown = 20;
    a.shut = true;
    a.shutTimer = 50;
    a.mouth = 1.5;
    play('poke_boredom', a.piece ? 0.75 : 2 ** ((actors().indexOf(el) * 1.5) / 12));
  }
  function stepAntic(a) {
    if (a.naked) {
      a.fuzzGone -= 0.005;
      if (a.fuzzGone <= 0) {
        a.fuzzGone = 0;
        a.naked = false;
      }
    }
    if (a.performing) performAntic(a);
    a.mouth = Math.max(0, a.mouth - 0.02);
    if (a.moveDown) a.moveDown = a.moveDown * 0.93 < 1 ? 0 : a.moveDown * 0.93;   // EaseSquash
    if (a.shut && --a.shutTimer <= 0) a.shut = false;
  }
  const RAD = Math.PI / 180;
  function performAntic(a) {
    const f = a.frame;
    switch (a.kind) {
      case 0:                                  // oggle: wincing, swaying from side to side
        a.wince = true;
        a.ox = Math.sin(f * 15 * RAD) * 3;
        if (f >= 60) a.wince = a.performing = false;
        break;
      case 1:                                  // cough, twice
        if (f === 0 || f === 40) {
          a.mouth = 1;
          play('cough');
        }
        a.wince = f < 20 || f > 40;
        if (f === 60) a.wince = a.performing = false;
        break;
      case 2:                                  // small sneeze: the face sinks while it builds up
        if (f > 30) {
          a.oy = 0;
          if (f === 31) a.mouth = 1;
          a.sneezing = false;
          a.wince = true;
        } else a.oy *= 0.97;
        if (f === 60) a.wince = a.performing = false;
        break;
      case 3:                                  // huge sneeze: blinks building up, then the fuzz flies off
        if (f <= 110) {
          a.oy *= 0.99;
          if (f === 50 || f === 70 || f === 90) {
            a.shut = true;
            a.shutTimer = 10;
          }
        } else {
          if (f === 111) blowFuzz(a);
          if (f < 120) shake = 20;
          a.oy = 0;
          a.sneezing = false;
          a.mouth = 1;
          a.wince = true;
        }
        if (f === 150) a.wince = a.performing = false;
        break;
      case 4:                                  // squeeze (the 2x2): a squeak at the end
        if (f === 50) {
          play('pokechuzzle', 0.75);
          a.mouth = 1;
          a.performing = false;
          a.shut = true;
          a.shutTimer = 25;
        }
        break;
      case 5: {                                // hurk: the cheeks puff in and out
        const c = Math.sin(f * 2 * RAD) * 4;
        a.scale = c > 1 ? 1 : Math.cos(f * 2 * RAD) * 4;
        if (f >= 90) a.cheeks = a.performing = false;
        break;
      }
      case 6: {                                // long hurk: the cheeks puff, then it spits it out
        const c = Math.abs(Math.sin(f * 1.75 * RAD) * 4);
        a.scale = c > 1 ? 1 : Math.abs(Math.cos(f * 1.75 * RAD) * 4);
        if (f === 90) {
          play('hurk');
          a.mouth = 2;
          a.shut = true;
          a.shutTimer = 70;
        }
        if (f >= 180) a.cheeks = a.performing = false;
        break;
      }
      case 7:                                  // the 2x2's beauty show
        if (f >= 110) a.beauty = a.performing = false;
        break;
    }
    a.frame++;
  }

  // The particles over the page (Animators.cpp), in CSS pixels at the chuzzle's scale: the
  // fuzz tufts (Animator_Fuzz), the splode strip, the stars and the shockwave of the 2x2's show.
  const fx = [];
  let shake = 0;                               // the page shaking (app->mShake), updates left
  const rnd = (a, b) => a + Math.random() * (b - a);
  function actorCentre(el) {
    const { node, units } = el._actor, r = el.getBoundingClientRect(), sc = r.width / units;
    const half = node._piece ? 80 : 40;
    return { x: r.left + (node._x0 + half) * sc, y: r.top + (node._y0 + half) * sc, sc };
  }
  function fuzz(x, y, sc, rgb) {
    return { t: 'fuzz', x, y, sc, rgb, rise: (rand(5) / 5 * 0.2 + 1) * 0.1, alpha: 1, fade: (rand(5) / 5 * 0.2 + 1) * 0.005, phase: rand(360), rot: 0, delay: 17 };
  }
  // the huge sneeze: 25 tufts round it and a ring of 180 flying out, the splode, and it is bald
  function blowFuzz(a) {
    const { x, y, sc } = actorCentre(a.el), rgb = BODY[a.k];
    a.fuzzGone = 4;
    a.naked = true;
    for (let i = 0; i < 25; i++) fx.push(fuzz(x + rnd(-50, 50) * sc, y + rnd(-50, 50) * sc, sc, rgb));
    for (let deg = 0; deg < 360; deg += 2) {
      fx.push(fuzz(x + (Math.sin(deg * RAD) * 75 + rnd(-40, 40)) * sc, y + (-Math.cos(deg * RAD) * 75 + rnd(-40, 40)) * sc, sc, rgb));
    }
    fx.push({ t: 'strip', x, y, sc, rgb, frame: 0, alpha: 1 });
  }
  // the beauty show: ten stars shooting out about 45 degrees apart, a shockwave, 200 tufts in a spiral
  function beautyBurst(a) {
    const { x, y, sc } = actorCentre(a.el), rgb = BODY[a.k];
    let angle = rand(360);
    for (let i = 0; i < 10; i++) {
      const t = angle * RAD, d = rand(100) + 50;
      fx.push({ t: 'star', x: x + Math.cos(t) * d * sc, y: y + Math.sin(t) * d * sc, vx: Math.cos(t) * 2, vy: Math.sin(t) * 2,
        sc, alpha: 1, fade: 0.01, scale: 0.5 + rand(10) / 20, rot: rnd(-25, 25) });
      angle += 45 + rnd(-10, 10);
    }
    fx.push({ t: 'wave', x, y, sc, alpha: 1, scale: 2, grow: (rand(5) / 5 * 0.01 + 0.05) * 0.5, fade: (rand(5) / 3 * 0.01 + 0.03) * 2 / 1.5 });
    for (let deg = 0; deg < 600; deg += 3) {
      fx.push(fuzz(x + (Math.sin(deg * RAD) * 160 + rnd(-40, 40)) * sc, y + (-Math.cos(deg * RAD) * 160 + rnd(-40, 40)) * sc, sc, rgb));
    }
  }
  function stepFx() {
    for (let i = fx.length - 1; i >= 0; i--) {
      const p = fx[i];
      if (p.t === 'fuzz') {
        if (p.delay > 0) {
          p.delay--;
          continue;
        }
        p.alpha -= p.fade;
        p.y += p.rise * p.sc;
        p.phase += 2;
        p.rot = Math.sin(p.phase * RAD) * 40;
      } else if (p.t === 'strip') {
        p.frame += 0.5;
        if (p.frame >= 17) p.alpha = 0;
      } else if (p.t === 'star') {
        p.alpha -= p.fade;
        p.x += p.vx * p.sc;
        p.y += p.vy * p.sc;
      } else if (p.t === 'wave') {
        p.alpha -= p.fade;
        p.scale += p.grow;
      }
      if (p.alpha <= 0) fx.splice(i, 1);
    }
    if (shake > 0) shake--;
  }
  function drawFx(ctx, dpr) {
    for (const p of fx) {
      if (p.t === 'fuzz' && p.delay > 0) continue;
      const s = p.sc * dpr;
      ctx.globalAlpha = Math.max(0, Math.min(1, p.alpha));
      if (p.t === 'fuzz') {
        const img = tinted('fuzz', p.rgb);
        if (!img) continue;
        ctx.setTransform(s, 0, 0, s, p.x * dpr, p.y * dpr);
        ctx.rotate(p.rot * RAD);
        ctx.drawImage(img, -8, -9);
      } else if (p.t === 'strip') {
        const img = tinted('splode', p.rgb);
        if (!img) continue;
        ctx.setTransform(s, 0, 0, s, p.x * dpr, p.y * dpr);
        ctx.drawImage(img, Math.floor(p.frame) * 149, 0, 149, 149, -74.5, -74.5, 149, 149);
      } else if (p.t === 'star') {
        const img = tinted('star', [1, 1, 0]);
        if (!img) continue;
        ctx.setTransform(s * p.scale, 0, 0, s * p.scale, p.x * dpr, p.y * dpr);
        ctx.rotate(p.rot * RAD);
        ctx.drawImage(img, -32, -32);
      } else if (p.t === 'wave' && SHEETS.shockwave) {
        ctx.setTransform(s * p.scale, 0, 0, s * p.scale, p.x * dpr, p.y * dpr);
        ctx.drawImage(SHEETS.shockwave, -128, -128);
      }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
  }

  // The chuzzles' voices (assets/sounds, from the game's Sounds/speaks), pitched like
  // PlayPitched by the playback rate.
  const sounds = new Map();
  function play(name, rate = 1) {
    let a = sounds.get(name);
    if (!a) {
      a = new Audio(`assets/sounds/${name}.ogg`);
      sounds.set(name, a);
    }
    const s = a.cloneNode();
    s.volume = 0.6;
    s.preservesPitch = false;
    s.playbackRate = rate;
    s.play().catch(() => {});
  }

  // The layer over the page: the held chuzzle, redrawn on every raw pointer event rather than
  // once a frame (a low-latency canvas where the browser allows one), so it keeps up with the
  // cursor; and the particles.
  const dragLayer = { cv: null, ctx: null, node: null, board: null, hc: null, hctx: null, hsig: null, fxWas: false };
  // The held chuzzle: a small canvas of its own over the page, moved by a CSS transform as the
  // pointer moves (the compositor shifts a texture, nothing is drawn) and drawn again only
  // when its pose changes. Before, a full-window canvas was cleared and drawn on every mouse
  // event, and the browser composed the whole window for it each time.
  function drawHeld() {
    const d = dragLayer, node = d.node;
    if (!node || !node.isConnected || !SHEETS.body) {
      if (d.hc) d.hc.style.display = 'none';
      d.hsig = null;
      return;
    }
    if (!d.hc) {
      d.hc = document.createElement('canvas');
      d.hc.className = 'drag-held';
      d.hc.setAttribute('aria-hidden', 'true');
      document.body.appendChild(d.hc);
      d.hctx = d.hc.getContext('2d');
    }
    const dpr = window.devicePixelRatio || 1, rect = d.board;
    const s = layout.u * dpr, cellPx = 80 * s, padPx = 10 * s, half = (node._piece ? 80 : 40) * s;
    const reach = Math.ceil(half * 1.12 + 30 * dpr), W = 2 * reach, H = 2 * reach + Math.ceil(12 * dpr);
    const x = rect.left * dpr + padPx + node._c * cellPx + (node._dx || 0) * dpr;
    const y = rect.top * dpr + padPx + node._r * cellPx + (node._dy || 0) * dpr;
    // the canvas sits on whole device pixels (a transform by a fraction would blur it)
    const x0 = Math.floor(x + half - reach), y0 = Math.floor(y + half - reach);
    if (d.hc.width !== W || d.hc.height !== H) {
      d.hc.width = W;
      d.hc.height = H;
      d.hc.style.width = `${W / dpr}px`;
      d.hc.style.height = `${H / dpr}px`;
      d.hsig = null;
    }
    d.hc.style.transform = `translate(${x0 / dpr}px, ${y0 / dpr}px)`;
    d.hc.style.display = '';
    const now = performance.now(), p = pose(node, now, 0, true);
    if (p.sig === d.hsig) return;
    d.hsig = p.sig;
    const ctx = d.hctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.imageSmoothingQuality = 'high';
    // the chuzzle at the same spot of its canvas whatever the pointer's fraction of a pixel
    ctx.translate(reach - half - x, reach - half - y);
    ctx.translate(rect.left * dpr, rect.top * dpr);
    drawChuzzle(ctx, node, p, now, 0, s, Math.round, cellPx, padPx, true);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // The particles (fuzz, stars, the shockwave): a full-window canvas, shown only while there
  // are any.
  function drawOverlay() {
    const d = dragLayer;
    if (!fx.length && !d.fxWas) return;
    if (!d.cv) {
      d.cv = document.createElement('canvas');
      d.cv.className = 'drag-layer';
      d.cv.setAttribute('aria-hidden', 'true');
      document.body.appendChild(d.cv);
      d.ctx = d.cv.getContext('2d', { desynchronized: true });
    }
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(innerWidth * dpr), H = Math.round(innerHeight * dpr);
    if (d.cv.width !== W || d.cv.height !== H) {
      d.cv.width = W;
      d.cv.height = H;
    }
    const ctx = d.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    d.fxWas = fx.length > 0;
    d.cv.style.display = fx.length ? '' : 'none';
    if (fx.length) drawFx(ctx, dpr);
  }

  // The Scrambulator's arrows (NoMoves::Draw) on the candy's orange ends: a 5-frame strip out
  // and back, a frame every 90 ms, one 120 left of the candy's centre and 3 down, one mirrored
  // 120 right. The canvas covers the Solve button (the candy, 320 x 202).
  function drawArrows(now) {
    const cv = $('solveArrows');
    if (!cv || !SHEETS.arrows) return;
    const step = Math.floor(now / 90) % 10, frame = step > 4 ? 9 - step : step;
    if (frame === cv._frame && now - (cv._measured || 0) < 1000) return;
    const dpr = window.devicePixelRatio || 1;
    if (now - (cv._measured || 0) >= 1000) {
      cv._w = cv.clientWidth;
      cv._h = cv.clientHeight;
      cv._measured = now;
    }
    if (!cv._w) return;
    const W = Math.round(cv._w * dpr), H = Math.round(cv._h * dpr);
    if (cv.width !== W || cv.height !== H) {
      cv.width = W;
      cv.height = H;
    } else if (frame === cv._frame) return;
    cv._frame = frame;
    const ctx = cv._ctx || (cv._ctx = cv.getContext('2d')), s = W / 320, R = Math.round;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const aw = R(60 * s), ah = R(104 * s), ax = R(W / 2 - 120 * s - 30 * s), ay = R(H / 2 + 3 * s - 52 * s);
    const arrow = cut('arrows', frame * 60, 0, 60, 104, aw, ah);
    blit(ctx, arrow, ax, ay);
    ctx.setTransform(-1, 0, 0, 1, W, 0);
    blit(ctx, arrow, ax, ay);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  // Chuzzles easing into their cells after a board is redrawn: [r, c, dx, dy, size], the
  // offset (CSS pixels) and size each starts from.
  function glide(el, list) {
    for (const [r, c, dx, dy, sc] of list) {
      const n = el.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
      if (n) Object.assign(n, { _ox: dx, _oy: dy, _sc: sc || 1 });
    }
  }

  // One loop, every frame: the game's 100 updates a second for the antics and particles, then
  // the boards (a board out of sight, on a phone, is skipped), the logo's and the palette's
  // chuzzles, the arrows and the layer over the page. With nothing but the idle sway going on
  // (no drag, no particles, the mouse still) the canvases are let change only every 80 ms.
  let lastFrame = 0, updates = 0, lastTick = 0;
  const DEBUG = self.MB_DEBUG = {};        // switches for measuring (the console / tests)
  function frameLoop(now) {
    const dt = lastFrame ? Math.min(100, now - lastFrame) : 16;
    lastFrame = now;
    eyeStamp++;
    updates = Math.min(updates + dt, 200);
    while (updates >= 10) {
      updates -= 10;
      for (const a of antics.values()) stepAntic(a);
      if (fx.length || shake) stepFx();
    }
    // idle: every canvas waits for the same 80 ms tick, so the page composes 12 frames a
    // second at most, not one per canvas that changed (a body frame lasts ~87 ms anyway; in
    // Firefox each composite, not the drawing, is what the idle page costs)
    // (with the match outline's stars running, idle still, but a tick every 20 ms: the stars
    // at 50 frames a second, the rest of the page with them, not one frame per screen refresh)
    const idle = !dragLayer.node && !fx.length && now - lastPointer > 400 && (DEBUG.idleMs == null || DEBUG.idleMs > 0);
    const tick = !idle || now - lastTick >= (starsAlive ? 20 : (DEBUG.idleMs ?? 80));
    if (tick) lastTick = now;
    const rest = starsAlive ? 16 : 40;
    starsAlive = false;
    if (!DEBUG.noBoards) for (const id of ['goalBoard', 'startBoard']) {
      if (layout.two || state.side === (id === 'goalBoard' ? 'goal' : 'start')) drawBoard($(id), now, dt, !tick);
    }
    if (!DEBUG.noActors) for (const el of actors()) drawActor(el, now, dt, !tick);
    if (!DEBUG.noArrows && tick) drawArrows(now);
    if (dragLayer.node) drawHeld();
    drawOverlay();
    const app = $('app');
    if (shake > 0) app.style.translate = `${rnd(-1, 1) * shake * 0.4}px ${rnd(-1, 1) * shake * 0.4}px`;
    else if (app.style.translate) app.style.translate = '';
    // idle, the loop itself rests between ticks (a frame callback every vsync is not free);
    // the held chuzzle is drawn by the pointer events, so a drag never waits for it
    if (idle) setTimeout(() => requestAnimationFrame(frameLoop), rest);
    else requestAnimationFrame(frameLoop);
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
  // The width fontText gives a text, in CSS pixels (0 until the font has loaded).
  function fontWidth(text, size) {
    if (!FONT) return 0;
    let w = 0;
    for (const ch of text) {
      const g = FONT.glyphs[ch === '≤' ? '<' : ch];
      w += ch === ' ' || !g ? FONT.space : g.w;
    }
    return Math.ceil(w * size / FONT.size + 2 * Math.max(1, Math.round(size / 9)) + 2);
  }
  function drawFontText(el) {
    const job = fontJobs.get(el);
    if (!job || !FONT || !fontAtlas.naturalWidth) return;
    const { text, size, fill, outline } = job;
    const s = size / FONT.size, dpr = Math.min(3, window.devicePixelRatio || 1);
    const o = Math.max(1, Math.round(size / 9));                       // outline width, CSS px
    let w = 0;
    const glyphOf = (c) => FONT.glyphs[c === '≤' ? '<' : c];
    for (const ch of text) w += ch === ' ' ? FONT.space : (glyphOf(ch) ? glyphOf(ch).w : FONT.space);
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
      const gl = glyphOf(c);
      if (gl) {
        const [rx, ry, rw, rh] = gl.r;
        const lift = c === '≤' ? FONT.ascent * 0.16 * s : 0;
        g.drawImage(fontAtlas, rx, ry, rw, rh, x + gl.o[0] * s, o + 1 + gl.o[1] * s - lift, rw * s, rh * s);
        if (c === '≤') {
          const bar = Math.max(1.5, FONT.ascent * 0.11 * s);
          g.fillRect(x + gl.o[0] * s, o + 1 + FONT.ascent * s - bar, rw * s, bar);
        }
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
  // (the bar's own width: phones turned sideways give it a column, not the window)
  const infoSize = () => Math.round(Math.max(15, Math.min(22, (document.querySelector('.subbar').clientWidth || innerWidth) / 42)));

  // ---------------------------------------------------------------- layout
  // The boards fill the space between the info bar and the dock: two side by side when that
  // gives at least ~3/4 of the size one board would get, else one with Start / Goal tabs.
  // --u (one game pixel) follows; a board is 535 of them.
  const layout = { u: 0.6, two: false };
  function layoutStage() {
    for (const id of ['thumbGoal', 'thumbStart']) {   // the tabs' little boards, in the new zoom's pixels
      const el = $(id);
      if (el._board && el._dpr !== (window.devicePixelRatio || 1)) renderThumb(el, el._board);
    }
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
  // Drawn on a canvas in whole device pixels (7 px dots a pixel apart at 1:1), because at a
  // fractional zoom a grid of 7 px elements lands on different device pixels row by row.
  let dotTints = null;
  function renderThumb(el, board) {
    const dpr = window.devicePixelRatio || 1;
    const t = Math.round(7 * dpr), gap = Math.max(1, Math.round(dpr)), pad = Math.round(3 * dpr), pitch = t + gap;
    const size = 2 * pad + 6 * pitch - gap;
    if (el.width !== size) {
      el.width = el.height = size;
      el.style.width = el.style.height = `${size / dpr}px`;
    }
    el._board = board;
    el._dpr = dpr;
    if (!dotTints) {
      const cs = getComputedStyle(document.documentElement);
      dotTints = COLORS.map((name, k) => cs.getPropertyValue(`--c${k}`).trim());
    }
    const ctx = el.getContext('2d');
    ctx.clearRect(0, 0, size, size);
    const f = board.fat;
    board.cells.forEach((v, k) => {
      const r = Math.floor(k / 6), c = k % 6;
      const piece = inPiece(f, r, c);
      if (piece && (r !== f.y || c !== f.x)) return;
      const w = piece ? 2 * pitch - gap : t, m = piece ? gap : 0;   // the big dot a pixel in from its cells
      ctx.fillStyle = dotTints[v];
      ctx.beginPath();
      ctx.arc(pad + c * pitch + w / 2, pad + r * pitch + w / 2, w / 2 - m, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  const colourCounts = (board) => {
    const n = Array(8).fill(0);
    for (const v of board.cells) n[v]++;
    return n;
  };

  function renderPalette() {
    const el = $('palette');
    if (!el.children.length) {
      COLORS.forEach((name, k) => {
        const s = document.createElement('button');
        s.type = 'button';
        s.className = 'swatch';
        s.setAttribute('role', 'radio');
        setChuzzle(s, k);
        attachActor(s, 'normal', `sw:${k}`);
        s.addEventListener('click', () => selectColour(k));
        el.appendChild(s);
      });
      const big = document.createElement('button');
      big.type = 'button';
      big.className = 'swatch big';
      big.id = 'fatSwatch';
      big.setAttribute('role', 'radio');
      attachActor(big, 'surface', 'sw:big');
      big.addEventListener('click', selectFatTool);
      el.appendChild(big);
    }
    COLORS.forEach((name, k) => {
      const s = el.children[k];
      s.setAttribute('aria-checked', String(state.tool === 'paint' && state.color === k));
      s.setAttribute('aria-label', name);
      s.title = `${name} (key ${k + 1})`;
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
    $('solveBtn').setAttribute('aria-label', game ? 'View the solutions' : 'Solve');
    paintOrb();
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
    state.tool = 'piece';          // (a colour swatch takes the brush back; a second click here
    if (state.mode === 'edit') state.side = 'goal';   // keeps it, so clicking on counts as pokes)
    render();
  }

  function setSide(side) {
    state.side = side;
    render();
  }

  // The Settings dialog: the solutions switch (the choice in gold, as in the header's switch)
  // and what it means. A change applies to the next search.
  function renderSettings() {
    for (const [id, on] of [['solsFirst', settings.first], ['solsAll', !settings.first]]) {
      $(id).setAttribute('aria-pressed', String(on));
      $(id).setAttribute('aria-checked', String(on));
      const label = $(id).querySelector('[data-label]');
      fontText(label, label.dataset.label, on ? { ...GOLD_TEXT, size: 16 } : { size: 16, fill: ['#d8d2f4', '#a9a2cf'] });
    }
    $('solsNote').textContent = settings.first
      ? 'Stops at the first solution of the shortest length. Every shorter length is still searched in full, so it is still the fewest moves, and long searches end many times sooner.'
      : 'Finds and counts every shortest solution. The last length is searched to the end, so long searches take several times longer.';
  }

  // The move limit beside Solve ("up to / 10 moves", with - and +). A game puzzle longer than
  // the page's search has no limit to set: Solve shows the desktop solver's answer.
  function renderDepth() {
    const d = state.maxDepth;
    $('depthBox').hidden = !!knownPreset();
    fontText($('depthText'), `≤ ${d} move${d > 1 ? 's' : ''}`, { size: 17 });
    const widest = fontWidth(`≤ ${MAX_DEPTH} moves`, 17);    // the plaque keeps one width
    $('depthText').parentElement.style.minWidth = widest ? `${widest + 12}px` : '';
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
    // spot: the cell under the pointer, when known (the 2x2 is one element whose row and
    // column are its top-left, so with the 2x2 tool its element alone would put it back in place)
    const paintAt = (cell, spot) => {
      const r = spot ? spot.r : +cell.dataset.r, c = spot ? spot.c : +cell.dataset.c;
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
    // The cells a drop would empty and the colour each would get: the held chuzzle's own cell
    // takes the one it lands on; the 2x2's freed cells take the chuzzles it would cover.
    const fillsFor = (t) => {
      if (!t) return null;
      const b = state[which];
      if (!t.piece) return [{ r: g.r, c: g.c, k: b.cells[t.r * 6 + t.c] }];
      const was = pieceCells(b.fat), now = pieceCells({ x: t.c, y: t.r });
      const freed = was.filter((k) => !now.includes(k)), taken = now.filter((k) => !was.includes(k));
      return freed.map((k, n) => ({ r: Math.floor(k / 6), c: k % 6, k: b.cells[taken[n]] }));
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
        paintAt(cell, spotAt(e.clientX, e.clientY));
      }
    });
    // the raw pointer stream (every mouse report, not once a frame) moves the held chuzzle
    el.addEventListener('pointerrawupdate', (e) => {
      if (!g || !g.drag) return;
      g.cell._dx = e.clientX - g.x;
      g.cell._dy = e.clientY - g.y;
      drawHeld();
    });
    if (which === 'goal') {
      el.addEventListener('pointerleave', () => { el._hover = null; });
    }
    el.addEventListener('pointermove', (e) => {
      if (which === 'goal') el._hover = e.pointerType === 'mouse' ? spotAt(e.clientX, e.clientY) : null;
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
        if (dragLayer.node !== g.cell) {
          dragLayer.node = g.cell;
          dragLayer.board = el.getBoundingClientRect();
        }
        if (!('onpointerrawupdate' in el)) drawHeld();
        const t = dropTarget(e);
        if (!g.target !== !t || (t && (t.r !== g.target.r || t.c !== g.target.c))) {
          el._fills = fillsFor(t);
          el._target = t;
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
        el._fills = null;
        el._target = null;
        dragLayer.node = null;
        drawHeld();
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
      // the one it swaps with flies over to fill the spot the held one left
      glides.push([from.r, from.c, (to.c - from.c) * P, (to.r - from.r) * P]);
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
    if (awayResult && onBoards(awayResult)) showResult(awayResult);
    if (!onBoards(run)) {
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

  // A custom puzzle's line: what to do first, then how far the start is from the goal (it
  // changes with every drag), then, once solved, the answer, worded as a game puzzle's is.
  function customInfo() {
    if (state.follows) return 'Paint the goal, then drag the start\'s Chuzzles into place';
    if (onBoards(result) && result.depth !== 0) {
      const r = result;
      return r.depth < 0 ? `Your puzzle  -  no solution in ${r.maxDepth} moves`
        : `Your puzzle  -  shortest is ${r.depth} move${r.depth > 1 ? 's' : ''}`
          + (r.first ? '' : `  -  ${r.count.toLocaleString()} solution${r.count > 1 ? 's' : ''}`);
    }
    const s = state.start, g = state.goal;
    let off = s.fat && g.fat && (s.fat.x !== g.fat.x || s.fat.y !== g.fat.y) ? 1 : 0;
    for (let i = 0; i < 36; i++) {
      const r = Math.floor(i / 6), c = i % 6;
      if (inPiece(s.fat, r, c)) continue;
      if (inPiece(g.fat, r, c) || s.cells[i] !== g.cells[i]) off++;
    }
    return off === 0 ? 'The start is already the goal' : `${off} Chuzzle${off > 1 ? 's' : ''} out of place`;
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
      text = 'Start and goal need the same Chuzzles: press Reset';
      warn = true;
    } else if (!p) {
      text = customInfo();
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
  // Chrome reports the RAM (navigator.deviceMemory, at most 8); Firefox and Safari report
  // nothing. A machine with 12 or more threads has the RAM to go with them as a rule, and
  // there the workers' copies of the goal side are what limits 11- and 12-move searches.
  function memoryBudget() {
    const dm = navigator.deviceMemory;
    if (dm && dm <= 2) return 400e6;
    if (dm && dm <= 4) return 800e6;
    return hardwareThreads >= 12 ? 2400e6 : 1600e6;
  }

  // One search at a time. It belongs to the boards it was started on (run.boards) and goes on
  // whatever puzzle is shown meanwhile: game puzzles can be viewed, other boards edited, and
  // the status line follows the search (a click on it goes back). A search that ends while
  // another puzzle is shown keeps its result (awayResult) until its puzzle is shown again.
  let run = null;
  let result = null;
  let awayResult = null;
  const onBoards = (r) => !!r && sameBoard(r.boards.start, state.start) && sameBoard(r.boards.goal, state.goal);

  const fmtTime = (s) => (s < 0.1 ? 'under 0.1 s' : s < 10 ? `${s.toFixed(1)} s` : s < 90 ? `${Math.round(s)} s`
    : `${Math.floor(s / 60)} min ${String(Math.round(s % 60)).padStart(2, '0')} s`);

  // The Solve orb: "Solve" ("View" for a game puzzle) when idle; while the puzzle shown is
  // being solved it fills from the bottom with the progress (or bobs when the step has none)
  // and shows the percentage; `label` is what the status line says the solver is doing. With
  // another puzzle shown during a search, the orb is that puzzle's: View works, Solve waits
  // for the search, and the cancel button stops the search.
  let stage = '', progress = 0;
  function setProgress(label, fraction) {
    stage = run ? label : '';
    progress = fraction;
    paintOrb();
  }
  function paintOrb() {
    const here = onBoards(run), known = !!knownPreset(), f = progress;
    fontText($('solveLabel'), !here ? (known ? 'View' : 'Solve') : f == null ? '...' : `${Math.floor(f * 100)}%`, ORB_TEXT);
    const fill = $('solveFill');
    fill.classList.toggle('indet', here && f == null);
    fill.style.transform = !here || f == null ? '' : `scaleY(${Math.round(f * 1000) / 1000})`;
    const btn = $('solveBtn'), stop = $('cancelBtn');
    btn.classList.toggle('running', here);
    btn.disabled = !!run && (here || !known);
    btn.title = run && !here && !known ? 'Another puzzle is being solved: one search at a time' : '';
    stop.hidden = !run;
    // a replay has nothing to solve (Edit goes back), unless a search can be stopped here
    $('solveBox').hidden = state.mode === 'replay' && !!result && !!result.replay && !run;
    stop.setAttribute('aria-label', here ? 'Cancel' : 'Stop solving the other puzzle');
    stop.title = here ? '' : 'Stop solving the other puzzle';
    $('depthBox').classList.toggle('busy', here);
  }

  // The status line as a link back to another puzzle (the one being solved, or solved).
  function statusLink(on) {
    const el = $('status');
    el.classList.toggle('link', on);
    if (on) {
      el.setAttribute('role', 'button');
      el.tabIndex = 0;
    } else {
      el.removeAttribute('role');
      el.removeAttribute('tabindex');
    }
  }

  // The status line when no search runs on the puzzle shown: the last result's summary while
  // the boards are still the ones it solved, or word of a search that ended on another puzzle.
  function idleStatus() {
    const el = $('status');
    if (onBoards(run) || el.classList.contains('error')) return;
    if (run) return;                 // a search of another puzzle: its timer writes the line
    if (awayResult) {
      const a = awayResult;
      el.textContent = `Your other puzzle is ${a.depth < 0 ? `searched: no solution in ${a.maxDepth} moves` : `solved: ${a.depth} moves`} · show it`;
      el.title = 'Back to the puzzle that was solved';
      statusLink(true);
      return;
    }
    statusLink(false);
    el.textContent = onBoards(result) ? result.summary : '';
    el.title = el.textContent;
  }

  // Back to the boards of a search (running, or ended while another puzzle was shown).
  function goToBoards(r) {
    endCoach();
    pushUndo();
    state.start = cloneBoard(r.boards.start);
    state.goal = cloneBoard(r.boards.goal);
    state.preset = '';
    state.follows = r.follows;
    state.picked = null;
    leaveReplay();
    changed({ keepPreset: true });
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

  // A game puzzle: every shortest solution, from solutions/ (the desktop solver's answers).
  async function viewKnown(p) {
    const r = { cancelled: false, t0: performance.now(), boards: { start: cloneBoard(state.start), goal: cloneBoard(state.goal) }, follows: false };
    try {
      const resp = await fetch(`solutions/${p.name}.txt`);
      if (!resp.ok) throw new Error(`could not load solutions/${p.name}.txt (${resp.status})`);
      const text = await resp.text();
      if (!onBoards(r)) return;         // another puzzle was opened meanwhile
      const count = text.split('\n').filter(Boolean).length;
      finish(r, { depth: p.moves, count, text, threads: 0, maxDepth: p.moves, known: true });
    } catch (err) {
      showError(`The solutions could not be loaded: ${err.message}`);
    }
  }

  async function solve() {
    if (state.coach) endCoach(true);
    const known = knownPreset();
    if (known) {
      viewKnown(known);
      return;
    }
    if (run) return;                    // one search at a time (the orb says so)
    const { spec, error } = problemSpec();
    if (error) {
      showError(error);
      return;
    }
    showError('');
    statusLink(false);
    awayResult = null;
    const maxDepth = state.maxDepth;
    const threads = settings.threads || hardwareThreads, first = settings.first;
    run = { workers: [], cancelled: false, t0: performance.now(), boards: { start: cloneBoard(state.start), goal: cloneBoard(state.goal) }, follows: state.follows };
    const thisRun = run;
    let detail = '';
    const showStatus = () => {
      const el = $('status'), away = !onBoards(thisRun);
      if (away && el.classList.contains('error')) return;     // about the puzzle shown
      statusLink(away);
      el.textContent = `${away ? 'Solving your other puzzle: ' : ''}${stage} · ${fmtTime((performance.now() - thisRun.t0) / 1000)}${detail}${away ? ' · show it' : ''}`;
      el.title = away ? 'Back to the puzzle being solved' : '';
    };
    const tick = setInterval(showStatus, 200);
    try {
      await workerUrlReady;
      const w0 = warm || new SolverWorker();
      warm = null;
      run.workers.push(w0);
      setProgress('Starting', null);
      showStatus();
      await w0.call('setup', { ...spec, first });
      setProgress(`Up to ${Math.min(6, maxDepth)} moves`, null);
      const small = await w0.call('small', { maxDepth });
      if (small.depth >= 0) {
        finish(thisRun, { depth: small.depth, count: small.count, text: small.text, threads: 1, maxDepth, first });
        return;
      }
      if (maxDepth <= 6) {
        finish(thisRun, { depth: -1, threads: 1, maxDepth });
        return;
      }
      setProgress('Building tables', null);
      const g = await w0.call('goal4');
      // Each worker holds the goal side; the join's class indexes about halve a worker's
      // search time but add to its memory, so they stay only while they still leave room
      // for at least half as many workers as without them.
      const fit = (bytes) => Math.max(1, Math.min(threads, Math.floor(memoryBudget() / (bytes * 1.5 + 32e6))));
      const nJoin = fit(g.memory), nPlain = fit(g.memory - g.classBytes);
      const join = 2 * nJoin >= nPlain;
      if (!join) await w0.call('join', { on: false });
      const n = join ? nJoin : nPlain;
      detail = ` · ${n} thread${n > 1 ? 's' : ''}`;
      for (let i = 1; i < n; i++) run.workers.push(new SolverWorker());
      await Promise.all(run.workers.slice(1).map((w) => w.call('setup', { ...spec, first }).then(() => w.call('join', { on: join })).then(() => w.call('goal4'))));
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
        // every worker's part, or (one solution) the first worker to find one: the others are
        // stopped with the run
        const res = await new Promise((resolve, reject) => {
          const out = [];
          let left = run.workers.length;
          run.workers.forEach((w, i) => w.call('depth', { depth: d }, (p) => { prog[i] = p; update(); }).then((r) => {
            out[i] = r;
            if (first && r.count > 0) resolve([r]);
            else if (--left === 0) resolve(out);
          }, reject));
        });
        const count = res.reduce((s, r) => s + r.count, 0);
        if (count > 0) {
          finish(thisRun, { depth: d, count, text: res.map((r) => r.text).join(''), threads: n, maxDepth, first });
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
      setProgress('', 0);
      statusLink(false);
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
    const done = { ...res, lines, secs, summary, boards: r.boards, follows: r.follows };
    if (!onBoards(r)) {
      awayResult = done;                // shown when its puzzle is (status line: "show it")
      return;
    }
    showResult(done);
  }

  function showResult(done) {
    const { secs, lines } = done, res = done;
    if (awayResult === done) {
      awayResult = null;
      statusLink(false);
    }
    result = { ...done, shown: 0, selected: -1, step: 0, playing: false };
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
      $('resultSub').textContent = (res.first ? 'The first shortest solution found (Settings can list all) · none shorter'
        : `${n.toLocaleString()} shortest solution${n > 1 ? 's' : ''} · none shorter`)
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
    renderInfo();
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
    // the plaque keeps the width of its longest label, so its arrows stay put
    const all = result.lines.length.toLocaleString(), label = $('solLabel');
    const widest = result.first ? 0 : fontWidth(`Solution ${all.replace(/\d/g, '8')} of ${all}`, infoSize());
    label.style.minWidth = widest ? `${widest + 16}px` : '';
    fontText(label, result.first ? 'Shortest solution' : `Solution ${(i + 1).toLocaleString()} of ${all}`, { size: infoSize() });
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
    $('threads').value = String(settings.threads);
    if ($('threads').selectedIndex < 0) $('threads').value = '0';
    $('threads').addEventListener('change', () => {
      settings.threads = +$('threads').value;
      saveSettings();
    });

    fontText($('brandMain'), 'Mind Bender', { size: 21 });
    fontText($('brandSub'), 'Solver', { ...GOLD_TEXT, size: 21 });
    setChuzzle($('logo'), 4);
    attachActor($('logo'), 'normal', 'logo');
    requestAnimationFrame(frameLoop);
    let lastMove = null;
    addEventListener('pointermove', (e) => {
      lastPointer = performance.now();
      if (!lastMove) requestAnimationFrame(() => { trackMouse(lastMove); followMouse(lastMove); lastMove = null; });
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
    fontText($('setTitleText'), 'Settings', { ...GOLD_TEXT, size: 26 });
    $('settingsBtn').addEventListener('click', () => $('settingsDialog').showModal());
    for (const [id, on] of [['solsFirst', true], ['solsAll', false]]) {
      $(id).addEventListener('click', () => {
        settings.first = on;
        saveSettings();
        renderSettings();
      });
    }
    renderSettings();
    fontText($('pzTitleText'), 'Game puzzles', { ...GOLD_TEXT, size: 26 });
    $('puzzlesBtn').addEventListener('click', () => $('puzzlesDialog').showModal());
    for (const d of [$('helpDialog'), $('puzzlesDialog'), $('settingsDialog')]) {
      d.addEventListener('click', (e) => { if (e.target === d) d.close(); });
    }

    $('solveBtn').addEventListener('click', solve);
    $('cancelBtn').addEventListener('click', cancel);
    const statusGo = () => {
      if (run && !onBoards(run)) goToBoards(run);
      else if (awayResult && !onBoards(awayResult)) goToBoards(awayResult);
    };
    $('status').addEventListener('click', statusGo);
    $('status').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        statusGo();
      }
    });
    $('moreBtn').addEventListener('click', showMore);
    $('solutionList').addEventListener('click', (e) => {
      const li = e.target.closest('li');
      if (!li) return;
      selectSolution(+li.dataset.i);
      scrollTo({ top: 0, behavior: 'smooth' });
    });
    $('editBtn').addEventListener('click', () => { leaveReplay(); render(); renderInfo(); });
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
