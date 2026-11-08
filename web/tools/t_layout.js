// Opens the puzzle in the URL and, with VIEW set in the page, its first solution; reports the
// dock's parts (for screenshots of the layout at several widths).
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
await sleep(400);
// ?pz=15-3 opens that game puzzle from the list first
const pz = new URLSearchParams(location.search).get('pz');
if (pz) {
  document.getElementById('puzzlesBtn').click();
  await sleep(300);
  [...document.querySelectorAll('#pzGrid button')].find((b) => b.title.startsWith(`Puzzle ${pz}:`)).click();
  await sleep(600);
}
if (pz || location.hash.includes('p=')) { document.getElementById('solveBtn').click(); await sleep(1500); }
const box = (sel) => { const e = document.querySelector(sel); if (!e || !e.offsetParent) return null; const b = e.getBoundingClientRect(); return [Math.round(b.left), Math.round(b.top), Math.round(b.width), Math.round(b.height)]; };
return { info: document.getElementById('info').getAttribute('aria-label'), dock: box('.dock'), track: box('.track'), chips: box('#replayChips'), solveBox: box('#solveBox'), nav: box('.sol-nav'), depth: box('#depthBox'), w: innerWidth };
