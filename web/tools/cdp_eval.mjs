// Runs a script (an async function body, from a file) in the page in headless Chrome and prints
// its JSON result. usage: node cdp_eval.mjs <url> <script.js> [--shot out.png]
import { spawn } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, file] = process.argv.slice(2);
const shotAt = process.argv.indexOf('--shot');
const port = 9333 + Math.floor(Math.random() * 500);
const dir = mkdtempSync(join(process.env.SCRATCH || tmpdir(), 'chr-'));
const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`,
  (process.env.WIN || '--window-size=1400,1000'), ...(process.env.EXTRA ? process.env.EXTRA.split(' ') : []), '--no-first-run', '--no-default-browser-check', 'about:blank',
], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); }
}
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
const logs = [];
ws.addEventListener('message', (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  if (msg.method === 'Runtime.exceptionThrown') logs.push(msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('Runtime.enable');
await send('Page.enable');
await send('Page.navigate', { url });
await sleep(1200);
await send('Runtime.evaluate', { expression: `localStorage.setItem('mb-tutorial','done'); location.reload(); 1` });
await sleep(1500);
const res = await send('Runtime.evaluate', { expression: `(async () => { ${readFileSync(file, 'utf8')} })()`, awaitPromise: true, returnByValue: true });
console.log(JSON.stringify(res.result?.result?.value ?? res.result, null, 1));
if (shotAt > 0) {
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(process.argv[shotAt + 1], Buffer.from(shot.result.data, 'base64'));
}
if (process.env.THEN) {
  const r2 = await send('Runtime.evaluate', { expression: `(async () => { ${readFileSync(process.env.THEN, 'utf8')} })()`, awaitPromise: true, returnByValue: true });
  console.log(JSON.stringify(r2.result?.result?.value ?? r2.result));
}
if (logs.length) console.log('EXCEPTIONS:', logs);
ws.close();
chrome.kill();
process.exit(0);
