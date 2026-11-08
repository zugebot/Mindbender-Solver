// A drag that never ends: a start chuzzle is lifted and moved round in a circle with a
// pointer move every frame (the page redraws the held chuzzle's layer each one).
const sb = document.getElementById('startBoard');
const cell = sb.querySelector('.cell[data-r="0"][data-c="1"]');
const q = cell.getBoundingClientRect(), sx = q.left + q.width / 2, sy = q.top + q.height / 2;
const ev = (type, x, y) => sb.dispatchEvent(new PointerEvent(type, { bubbles: true, clientX: x, clientY: y, button: 0, pointerId: 1, isPrimary: true, pointerType: 'mouse' }));
ev('pointerdown', sx, sy);
const t0 = performance.now();
function tick() {
  const t = (performance.now() - t0) / 1000;
  ev('pointermove', sx + 120 + Math.cos(t * 2) * 100, sy + 120 + Math.sin(t * 2) * 100);
  requestAnimationFrame(tick);
}
tick();
return 1;
