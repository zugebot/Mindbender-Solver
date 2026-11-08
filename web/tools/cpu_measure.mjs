// Loads a page in headless Chrome, lets it idle, then measures the CPU time every Chrome
// process of this instance uses over N seconds (like Task Manager), by process type, as a
// percentage of one core. usage: node cpu_measure.mjs <url> [seconds] [--js "expr run before measuring"]
// Serve the page first: python -m http.server 8770 --directory web
import { spawn, execFileSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const url = process.argv[2];
const secs = +(process.argv[3] || 5);
const jsAt = process.argv.indexOf('--js');
const port = 9333 + Math.floor(Math.random() * 500);
const dir = mkdtempSync(join(process.env.SCRATCH || tmpdir(), 'chr-'));
const chrome = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe', [
  '--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${dir}`,
  ...(process.env.CHROME_FLAGS ? process.env.CHROME_FLAGS.split(' ') : []),
  '--window-size=1600,1000', '--no-first-run', '--no-default-browser-check', 'about:blank',
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
ws.addEventListener('message', (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evalJs = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send('Page.enable');
await send('Page.navigate', { url });
await sleep(1500);
if (!url.startsWith('about:')) {
  await evalJs(`localStorage.setItem('mb-tutorial','done'); location.reload(); 1`);
  await sleep(2500);
}
if (jsAt > 0) { await evalJs(process.argv[jsAt + 1]); await sleep(500); }
const gpu = await evalJs(`(() => { try { const c = document.createElement('canvas'); const g = c.getContext('webgl'); const d = g.getExtension('WEBGL_debug_renderer_info'); return g.getParameter(d.UNMASKED_RENDERER_WEBGL); } catch (e) { return String(e); } })()`);
await evalJs(`window.__f = 0; (function t(){ window.__f++; requestAnimationFrame(t); })(); 1`);

const sample = () => {
  const out = execFileSync('powershell.exe', ['-NoProfile', '-Command',
    `Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" | Where-Object { $_.ProcessId -eq ${chrome.pid} -or $_.ParentProcessId -eq ${chrome.pid} } | ForEach-Object { $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; if ($p) { $t = 'browser'; if ($_.CommandLine -match '--type=([a-z-]+)') { $t = $Matches[1] }; "$($_.ProcessId)|$t|$($p.TotalProcessorTime.TotalMilliseconds)" } }`]).toString().trim().split(/\r?\n/).filter(Boolean);
  const m = new Map();
  for (const line of out) { const [pid, type, ms] = line.split('|'); m.set(pid, { type, ms: +ms }); }
  return m;
};
const a = sample();
const f0 = await evalJs('window.__f');
const t0 = performance.now();
await sleep(secs * 1000);
const b = sample();
const f1 = await evalJs('window.__f');
const el = (performance.now() - t0) / 1000;
const byType = {};
for (const [pid, v] of b) {
  const prev = a.get(pid);
  const d = v.ms - (prev ? prev.ms : 0);
  byType[v.type] = +((byType[v.type] || 0) + d / el / 10).toFixed(1);   // % of one core
}
const total = +Object.values(byType).reduce((x, y) => x + y, 0).toFixed(1);
console.log(JSON.stringify({ url, gpu, seconds: +el.toFixed(1), fps: +((f1 - f0) / el).toFixed(0), coresPct: byType, totalOneCorePct: total }));
ws.close();
chrome.kill();
process.exit(0);
