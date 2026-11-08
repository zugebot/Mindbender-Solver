const sleep = (ms) => new Promise(r => setTimeout(r, ms));
document.getElementById('solveBtn').click();
for (let i = 0; i < 100 && !document.querySelector('#replayChips button'); i++) await sleep(100);
await sleep(300);
const text = document.getElementById('stepText').innerText;
const d = document.getElementById('guidePath').getAttribute('d');
document.getElementById('nextBtn').click();
await sleep(+(new URLSearchParams(location.search).get('t') || 650));
return { text, arrow: d, fat: document.querySelector('#startBoard .cell.piece')?.dataset };
