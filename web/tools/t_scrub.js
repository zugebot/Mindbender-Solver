// A replay-style lane scrub that never ends: row 2 of the start board slides back and forth
// every frame (the page redraws the whole board each time), with pointer moves so the page
// counts as active.
const sb = document.getElementById('startBoard');
const cells = [...sb.querySelectorAll('.cell[data-r="2"]')];
const t0 = performance.now();
let n = 0;
function tick() {
  const p = (performance.now() - t0) / 1000;
  for (const c of cells) { c._off = Math.sin(p * 3) * 60; c._col = false; c._f = 25; }
  if (++n % 2 === 0) document.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: 300 + (n % 50), clientY: 300, pointerType: 'mouse' }));
  requestAnimationFrame(tick);
}
tick();
return cells.length;
