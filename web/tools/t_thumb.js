const sleep = (ms) => new Promise(r => setTimeout(r, ms));
await sleep(300);
const r = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return [b.left, b.top, b.width, b.height, document.getElementById(id).width]; };
return { goal: r('thumbGoal'), start: r('thumbStart'), dpr: devicePixelRatio, one: document.getElementById('stage').className };
