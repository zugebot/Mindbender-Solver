// with the 2x2 tool, click each cell of the goal's 2x2: the piece must move so its top-left is
// the clicked cell (kept inside the board)
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const gb = document.getElementById('goalBoard');
const fire = (type, x, y) => gb.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0, pointerId: 1, isPrimary: true, pointerType: 'mouse' }));
document.getElementById('fatSwatch').click();
const out = [];
for (const [dr, dc] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
  const p = gb.querySelector('.cell.piece');
  const r0 = +p.dataset.r, c0 = +p.dataset.c, b = p.getBoundingClientRect(), h = b.width / 2;
  const x = b.left + h * (dc + 0.5), y = b.top + h * (dr + 0.5);
  fire('pointerdown', x, y); fire('pointerup', x, y);
  await sleep(80);
  const q = gb.querySelector('.cell.piece');
  out.push(`clicked (${r0 + dr},${c0 + dc}) of piece at (${r0},${c0}) -> piece at (${q.dataset.r},${q.dataset.c})`);
}
return out;
