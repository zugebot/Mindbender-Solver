// web/screenshot.js - reads a Mind Bender puzzle from a screenshot of the game: the board (the
// start) and the small goal picture beside it. Works on any of the games' layouts (Chuzzle 1,
// Chuzzle 2, mobile) at any size: it looks for the two 6x6 grids of Chuzzles rather than for
// fixed places on the screen.
//
// How: every pixel gets a colour class (the eight Chuzzle colours by hue, white, or none). A
// 6x6 grid of pitch p at (x, y) scores well when the middle of each cell is mostly one colour
// and the points where four cells meet are dark (the gaps between Chuzzles). The search runs
// on halved copies of the picture, pitches 12-24 pixels on each (8-24 on a small picture's full size),
// so every grid size costs about the same. The best grid is the board; the goal is the best grid 2.5-6 times smaller beside it.
// Each cell is then read at full size: its most common colour, white only when no colour
// covers a fair part of it (Chuzzles have white eyes). Text drawn over the goal (Chuzzle 2's
// flashing "MATCH THIS") is pink and pale, so it reads as no colour and only hides pixels.
'use strict';

(function (root) {
  const NONE = 8, WHITE = 6;
  // hue ranges (degrees) of the coloured Chuzzles, in the solver's colour order: red, green,
  // blue, orange, yellow, purple, white, cyan
  const HUES = [[0, 15, 0], [15, 45, 3], [45, 90, 4], [90, 160, 1], [160, 215, 7], [215, 270, 2],
    [270, 330, 5], [330, 361, 0]];

  // label and gap flag of every colour, 5 bits a channel: bits 0-3 the class, bit 4 a gap
  const LUT = new Uint8Array(32768);
  for (let i = 0; i < 32768; i++) {
    const r = ((i >> 10) & 31) * 8 + 4, g = ((i >> 5) & 31) * 8 + 4, b = (i & 31) * 8 + 4;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const v = mx / 255, s = mx ? (mx - mn) / mx : 0;
    let label = NONE;
    // dim colours are backgrounds (Chuzzle 1's navy goal glass, Chuzzle 2's dark red space)
    if (v >= 0.5) {
      if (s < 0.22) label = v > 0.66 ? WHITE : NONE;
      else if (s >= 0.4) {
        let h;
        const d = mx - mn;
        if (mx === r) h = ((g - b) / d + 6) % 6;
        else if (mx === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h *= 60;
        for (const [lo, hi, k] of HUES) if (h >= lo && h < hi) { label = k; break; }
      }
    }
    LUT[i] = label | (v < 0.42 ? 16 : 0);
  }
  const lutIndex = (r, g, b) => ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);

  // ---------------------------------------------------------------- the picture, smaller

  function half(img) {
    const w = img.w >> 1, h = img.h >> 1, src = img.data, out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const a = ((2 * y) * img.w + 2 * x) * 4, b = a + img.w * 4, o = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) out[o + c] = (src[a + c] + src[a + 4 + c] + src[b + c] + src[b + 4 + c] + 2) >> 2;
        out[o + 3] = 255;
      }
    }
    return { data: out, w, h };
  }

  function labels(img) {
    const n = img.w * img.h, out = new Uint8Array(n), d = img.data;
    for (let i = 0; i < n; i++) out[i] = LUT[lutIndex(d[4 * i], d[4 * i + 1], d[4 * i + 2])];
    return out;
  }

  // the count of mask pixels in the b x b box centred on each pixel (clipped)
  function boxCount(mask, w, h, b, tmp, out) {
    const r0 = b >> 1, r1 = b - r0;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      let s = 0;
      for (let x = 0; x < Math.min(r1, w); x++) s += mask[row + x];
      for (let x = 0; x < w; x++) {
        tmp[row + x] = s;
        if (x + r1 < w) s += mask[row + x + r1];
        if (x >= r0) s -= mask[row + x - r0];
      }
    }
    const up = r0 * w, down = r1 * w, n = w * h;
    out.fill(0, 0, w);
    for (let i = 0; i < Math.min(down, n); i++) out[i % w] += tmp[i];
    for (let i = w; i < n; i++) {
      let s = out[i - w];
      if (i - w + down < n) s += tmp[i - w + down];
      if (i - w >= up) s -= tmp[i - w - up];
      out[i] = s;
    }
  }

  // one 0/1 mask per colour and one of the dark pixels
  function masks(lab) {
    const n = lab.length, out = [];
    for (let k = 0; k < 9; k++) out.push(new Uint8Array(n));
    for (let i = 0; i < n; i++) {
      const v = lab[i], k = v & 15;
      if (k < 8) out[k][i] = 1;
      if (v & 16) out[8][i] = 1;
    }
    return out;
  }

  // per pixel: the share of the b x b box's pixels in its most common colour
  function bestMap(mk, w, h, b) {
    const n = w * h, tmp = new Uint16Array(n), cnt = new Uint16Array(n), most = new Uint16Array(n);
    for (let k = 0; k < 8; k++) {
      boxCount(mk[k], w, h, b, tmp, cnt);
      for (let i = 0; i < n; i++) if (cnt[i] > most[i]) most[i] = cnt[i];
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = most[i] / (b * b);
    return out;
  }

  // per pixel: the share of dark pixels in the b x b box
  function darkMap(mk, w, h, b) {
    const n = w * h, tmp = new Uint16Array(n), cnt = new Uint16Array(n), out = new Float32Array(n);
    boxCount(mk[8], w, h, b, tmp, cnt);
    for (let i = 0; i < n; i++) out[i] = cnt[i] / (b * b);
    return out;
  }

  // The maps a pitch uses: a cell's box about half the pitch, a meeting point's a fifth. The
  // sizes are rounded to a few so the pitches of one level share them.
  function mapCache(mk, w, h) {
    const best = new Map(), dark = new Map();
    return (p) => {
      const b = 2 * Math.round(p * 0.25), bc = Math.max(2, Math.round(p * 0.2));
      if (!best.has(b)) best.set(b, bestMap(mk, w, h, b));
      if (!dark.has(bc)) dark.set(bc, darkMap(mk, w, h, bc));
      return { best: best.get(b), dark: dark.get(bc) };
    };
  }

  // ---------------------------------------------------------------- finding the grids

  function score(m, w, x0, y0, p) {
    let cells = 0, corners = 0;
    for (let i = 0; i < 6; i++) {
      const cy = Math.round(y0 + (i + 0.5) * p) * w;
      for (let j = 0; j < 6; j++) cells += m.best[cy + Math.round(x0 + (j + 0.5) * p)];
    }
    for (let i = 1; i < 6; i++) {
      const cy = Math.round(y0 + i * p) * w;
      for (let j = 1; j < 6; j++) corners += m.dark[cy + Math.round(x0 + j * p)];
    }
    return (cells / 36) * (0.25 + 0.75 * corners / 25);
  }

  // the best grids of one level: coarse steps of p/4, then each kept one to the pixel
  function searchLevel(img, scale) {
    const mk = masks(labels(img)), w = img.w, h = img.h, found = [];
    const pitches = [];
    // (down to 8 pixels on the full size of a small picture: a small window's goal)
    for (let p = scale === 1 && w * h <= 1.2e6 ? 8 : 12; p < 24; p *= 1.06) pitches.push(p);
    const mapsFor = mapCache(mk, w, h);
    for (const p of pitches) {
      if (6 * p >= Math.min(w, h)) break;
      const m = mapsFor(p), step = p / 4, top = [];
      for (let y0 = 0; y0 + 6 * p < h - 1; y0 += step) {
        for (let x0 = 0; x0 + 6 * p < w - 1; x0 += step) {
          const s = score(m, w, x0, y0, p);
          if (s > 0.3) top.push({ s, x: x0, y: y0, p });
        }
      }
      top.sort((a, b) => b.s - a.s);
      const kept = [];
      for (const c of top) {
        if (kept.length >= 4) break;
        if (kept.some((k) => Math.abs(k.x - c.x) < 2 * p && Math.abs(k.y - c.y) < 2 * p)) continue;
        kept.push(c);
      }
      for (const c of kept) {
        let bestC = c;
        for (const pp of [p / 1.03, p, p * 1.03]) {
          const mm = mapsFor(pp);
          for (let dy = -step; dy <= step; dy++) {
            for (let dx = -step; dx <= step; dx++) {
              const x0 = c.x + dx, y0 = c.y + dy;
              if (x0 < 0 || y0 < 0 || x0 + 6 * pp >= w - 1 || y0 + 6 * pp >= h - 1) continue;
              const s = score(mm, w, x0, y0, pp);
              if (s > bestC.s) bestC = { s, x: x0, y: y0, p: pp };
            }
          }
        }
        found.push({ s: bestC.s, x: bestC.x * scale, y: bestC.y * scale, p: bestC.p * scale });
      }
    }
    return found;
  }

  function findGrids(img) {
    let level = img, scale = 1, all = [];
    // the full-size level only for pictures up to ~2.5 MP (its grids are small: 8-24 pixels)
    while (level.w >= 72 && level.h >= 72) {
      if (level.w * level.h <= 2.6e6) all = all.concat(searchLevel(level, scale));
      level = half(level);
      scale *= 2;
    }
    all.sort((a, b) => b.s - a.s);
    // one grid per place: the best of any that overlap
    const grids = [];
    const overlap = (a, b) => {
      const ix = Math.min(a.x + 6 * a.p, b.x + 6 * b.p) - Math.max(a.x, b.x);
      const iy = Math.min(a.y + 6 * a.p, b.y + 6 * b.p) - Math.max(a.y, b.y);
      return ix > 0 && iy > 0 ? (ix * iy) / Math.min(36 * a.p * a.p, 36 * b.p * b.p) : 0;
    };
    // (only grids of about the same pitch: a grid at half a goal's pitch, each tile split in
    // four, can outscore the goal itself)
    const alike = (a, b) => Math.max(a.p / b.p, b.p / a.p) < 1.4;
    for (const g of all) if (!grids.some((k) => alike(k, g) && overlap(k, g) > 0.25)) grids.push(g);
    // the board and the goal: the best pair with the goal 2.5-6 times smaller
    let pair = null;
    for (const b of grids.slice(0, 12)) {
      for (const g of grids.slice(0, 24)) {
        const r = b.p / g.p;
        if (r < 2.5 || r > 6 || overlap(b, g) > 0) continue;
        const s = b.s + g.s;
        if (!pair || s > pair.s) pair = { s, board: b, goal: g };
      }
    }
    return { pair, grids };
  }

  // ---------------------------------------------------------------- reading the cells

  function counts(img, x0, y0, x1, y1) {
    const c = new Array(9).fill(0), d = img.data;
    let n = 0;
    x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
    x1 = Math.min(img.w, Math.round(x1)); y1 = Math.min(img.h, Math.round(y1));
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const i = (y * img.w + x) * 4;
        c[LUT[lutIndex(d[i], d[i + 1], d[i + 2])] & 15]++;
        n++;
      }
    }
    return { c, n };
  }

  // A cell's colour: the most common colour if it covers 18% of the cell's middle, else white
  // if that covers 25%; sure when the runner-up has under half as many pixels.
  function readCell(img, g, r, col) {
    const { c, n } = counts(img, g.x + (col + 0.2) * g.p, g.y + (r + 0.2) * g.p,
      g.x + (col + 0.8) * g.p, g.y + (r + 0.8) * g.p);
    const order = [0, 1, 2, 3, 4, 5, 7].sort((a, b) => c[b] - c[a]);
    const k = order[0], second = c[order[1]];
    if (c[k] >= 0.18 * n) return { k, sure: second < 0.5 * c[k] && c[k] >= 0.25 * n };
    if (c[WHITE] >= 0.25 * n) return { k: WHITE, sure: c[k] < 0.08 * n };
    return { k: c[k] > c[WHITE] ? k : WHITE, sure: false };
  }

  // The 2x2 Chuzzle: four cells of one colour whose meeting point is that colour too (between
  // four separate Chuzzles it is a dark gap).
  function readFat(img, g, cells) {
    let best = null;
    for (let r = 0; r < 5; r++) {
      for (let c = 0; c < 5; c++) {
        const k = cells[r * 6 + c];
        if ([r * 6 + c + 1, r * 6 + c + 6, r * 6 + c + 7].some((i) => cells[i] !== k)) continue;
        const cx = g.x + (c + 1) * g.p, cy = g.y + (r + 1) * g.p, h = g.p * 0.12;
        const t = counts(img, cx - h, cy - h, cx + h, cy + h);
        const share = t.c[k] / t.n;
        if (share > 0.6 && (!best || share > best.share)) best = { x: c, y: r, share };
      }
    }
    return best ? { x: best.x, y: best.y } : null;
  }

  function readBoard(img, g) {
    const cells = [], unsure = [];
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        const cell = readCell(img, g, r, c);
        cells.push(cell.k);
        if (!cell.sure) unsure.push(r * 6 + c);
      }
    }
    return { cells, fat: readFat(img, g, cells), unsure };
  }

  const tally = (b) => {
    const n = new Array(8).fill(0);
    b.cells.forEach((k) => n[k]++);
    return n;
  };

  // img: { data: RGBA bytes, w, h }. Returns { start, goal, grids, problem }: start and goal
  // { cells (36 colour indices, rows from the top), fat ({x, y} of the 2x2's top-left) or null,
  // unsure (cell indices read with doubt) }; problem: a sentence when the reading can't be
  // trusted as a puzzle (null when it can).
  function read(img) {
    const { pair } = findGrids(img);
    if (!pair) return { problem: 'No Mind Bender board found in this picture.' };
    const start = readBoard(img, pair.board), goal = readBoard(img, pair.goal);
    const out = { start, goal, grids: { board: pair.board, goal: pair.goal }, problem: null };
    const a = tally(start), b = tally(goal);
    if (goal.cells.every((k) => k === 1) && start.cells.some((k) => k !== 1)) {
      out.problem = 'This picture shows the puzzle solved (its goal turns green). Use one from before the last move.';
    } else if (!!start.fat !== !!goal.fat) {
      out.problem = 'This doesn\'t read as a Mind Bender puzzle: only one of its two boards seems to have a 2x2 Chuzzle.';
    } else if (a.some((n, k) => n !== b[k])) {
      out.problem = 'This doesn\'t read as a Mind Bender puzzle: the board and the goal don\'t hold the same Chuzzles.';
    }
    return out;
  }

  // the two parts of a grid's score at full size (for tests)
  function scoreParts(img, x0, y0, p) {
    const w = img.w, m = mapCache(masks(labels(img)), w, img.h)(p);
    const cells = [], corners = [];
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) cells.push(+m.best[Math.round(y0 + (i + 0.5) * p) * w + Math.round(x0 + (j + 0.5) * p)].toFixed(2));
    for (let i = 1; i < 6; i++) for (let j = 1; j < 6; j++) corners.push(+m.dark[Math.round(y0 + i * p) * w + Math.round(x0 + j * p)].toFixed(2));
    return { cells, corners, s: score(m, w, x0, y0, p) };
  }

  const api = { read, findGrids, readBoard, scoreParts, LUT, lutIndex };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MBScreenshot = api;
  // started as a worker (app.js): one picture in ({ w, h, buf } of RGBA bytes), its reading out
  if (typeof document === 'undefined' && typeof root.postMessage === 'function' && !(typeof module === 'object' && module.exports)) {
    root.onmessage = (e) => {
      const { w, h, buf } = e.data;
      let out;
      try {
        out = read({ data: new Uint8ClampedArray(buf), w, h });
      } catch (err) {
        out = { problem: `Could not read this picture (${err.message}).` };
      }
      root.postMessage(out);
    };
  }
})(typeof self !== 'undefined' ? self : this);
