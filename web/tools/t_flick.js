// lift the 2x2 and flick it between two spots on the far side for a while: the chain round
// the spot it left must keep running (a long trail), not start again at every flick
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const q = new URLSearchParams(location.search);
const cellOf = (b, r, c) => document.querySelector(`#${b}Board .cell[data-r="${r}"][data-c="${c}"]`);
const ctr = (el) => { const b = el.getBoundingClientRect(); return [b.left + b.width / 2, b.top + b.height / 2]; };
const fire = (b, type, x, y) => document.getElementById(b + 'Board').dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0, pointerId: 1, isPrimary: true, pointerType: 'mouse' }));
const piece = document.querySelector('#startBoard .cell.piece');
const [px, py] = ctr(piece), cw = cellOf('start', 0, 0).getBoundingClientRect().width;
fire('start', 'pointerdown', px, py);
for (let i = 1; i <= 8; i++) { fire('start', 'pointermove', px + cw * 3 * 0.125 * i, py + cw * 2 * 0.125 * i); await sleep(20); }
const t0 = performance.now();
let n = 0;
while (performance.now() - t0 < +(q.get('ms') || 1500)) { fire('start', 'pointermove', px + cw * (3 + (n++ % 2)), py + cw * 2); await sleep(+(q.get('gap') || 100)); }
return { flicks: n };
