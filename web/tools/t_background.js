// Start a search of the custom board in the link, open game puzzle 16-4 while it runs, View
// its solutions, wait for the search to end over there, then go back through the status line.
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const $ = (id) => document.getElementById(id);
const snap = (what) => ({ what, orb: $('solveLabel').getAttribute('aria-label') || $('solveLabel').textContent, running: $('solveBtn').classList.contains('running'),
  disabled: $('solveBtn').disabled, cancelHidden: $('cancelBtn').hidden, status: $('status').textContent, link: $('status').classList.contains('link'),
  info: $('info').textContent.slice(0, 60), resultTitle: $('resultTitle').getAttribute('aria-label') || '', resultSub: $('resultSub').textContent,
  mode: document.body.className });
const out = [];
$('solveBtn').click();
await sleep(1500);
out.push(snap('custom, searching'));
$('puzzlesBtn').click();
await sleep(200);
document.querySelector('.pz[data-name="16-4"]').click();
await sleep(300);
out.push(snap('16-4 opened during the search'));
$('solveBtn').click();               // View 16-4's stored solutions
await sleep(1200);
out.push(snap('16-4 viewed'));
for (let i = 0; i < 1200 && !$('cancelBtn').hidden; i++) await sleep(100);
await sleep(300);
out.push(snap('search ended while on 16-4'));
$('status').click();                 // back to the solved puzzle
await sleep(800);
out.push(snap('back through the status line'));
return out;
