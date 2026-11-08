const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const q = new URLSearchParams(location.search), mode = q.get('m') || 'hover', t = +(q.get('t') || 500);
const cellOf = (b, r, c) => document.querySelector(`#${b}Board .cell[data-r="${r}"][data-c="${c}"]`);
const ctr = (el) => { const b = el.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; };
const fire = (b, type, x, y) => document.getElementById(b + 'Board').dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0, pointerId: 1, isPrimary: true, pointerType: 'mouse' }));
const sw = [...document.querySelectorAll('#palette .swatch')];
sw[0].click(); document.getElementById('fatSwatch').click();
const piece = document.querySelector('#startBoard .cell.piece');
const [px, py] = ctr(piece), cw = cellOf('start', 0, 0).getBoundingClientRect().width;
if (mode === 'hover' || mode === 'gone') {
  const [x, y] = ctr(cellOf('goal', 3, 3));
  fire('goal', 'pointermove', x, y);
  fire('start', 'pointerdown', px, py); fire('start', 'pointerup', px, py);
  await sleep(t);
  if (mode === 'gone') { fire('goal', 'pointerleave', 0, 0); fire('goal', 'pointermove', 0, 0); fire('start', 'pointerdown', px, py); fire('start', 'pointerup', px, py); await sleep(1500); }
} else if (mode === 'drag') {
  fire('start', 'pointerdown', px, py);
  for (let i = 1; i <= 8; i++) { fire('start', 'pointermove', px + cw * 0.125 * i, py + (q.get('dy') ? cw * 0.125 * i : 0)); await sleep(20); }
  await sleep(t);
}
return { mode, hover: document.getElementById('goalBoard')._hover, picked: !!document.querySelector('#startBoard .cell.picked') };
