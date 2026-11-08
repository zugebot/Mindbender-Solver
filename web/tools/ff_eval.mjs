// Loads a page in headless Firefox (a fresh profile, driven over WebDriver BiDi), runs a script
// file in it (an async function body; `return` gives the result, printed as JSON), optionally
// evaluates the JS env var, then measures the CPU time of the whole Firefox process tree over
// N seconds as a percentage of one core (the way Task Manager counts). Headless Firefox
// composes in software, so its numbers are a relative measure, not what a GPU build costs.
// usage: node ff_eval.mjs <url> [script.js] [--secs N] [--shot out.png]
//   HEADED=1 opens a real window instead (GPU compositing, as the user's Firefox: the numbers
//   Task Manager would show, per core)
//   JS="Object.assign(MB_DEBUG, {noBoards: 1})" node ff_eval.mjs http://127.0.0.1:8770/ --secs 10
//   PROF=C:/tmp/prof.json node ff_eval.mjs ... --secs 6   records a Gecko profile of the whole
//   run (every thread, 2 ms samples) and writes it when Firefox closes; open it at
//   https://profiler.firefox.com/ (the frames come out unsymbolicated: thread shares only)
// Serve the page first: python -m http.server 8770 --directory web
import { spawn, execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { connect } from 'node:net';

const url = process.argv[2];
const file = process.argv[3] && !process.argv[3].startsWith('--') ? process.argv[3] : null;
const secsAt = process.argv.indexOf('--secs');
const secs = secsAt > 0 ? +process.argv[secsAt + 1] : 0;
const shotAt = process.argv.indexOf('--shot');
const port = 9500 + Math.floor(Math.random() * 500);
const dir = mkdtempSync(join(process.env.SCRATCH || tmpdir(), 'ffp-'));
writeFileSync(join(dir, 'user.js'), [
  'user_pref("browser.shell.checkDefaultBrowser", false);',
  'user_pref("app.update.enabled", false);',
  'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
  'user_pref("browser.startup.homepage_override.mstone", "ignore");',
  'user_pref("toolkit.telemetry.reportingpolicy.firstRun", false);',
  ].join('\n'));
const ff = spawn('C:/Program Files/Mozilla Firefox/firefox.exe', [
  ...(process.env.HEADED ? [] : ['-headless']), '-no-remote', '-profile', dir, '--remote-debugging-port', String(port),
  '-width', '1600', '-height', '1000', 'about:blank',
], { stdio: 'ignore', env: process.env.PROF ? { ...process.env, MOZ_PROFILER_STARTUP: '1', MOZ_PROFILER_SHUTDOWN: process.env.PROF, MOZ_PROFILER_STARTUP_FEATURES: 'js,stackwalk,cpu,screenshots', MOZ_PROFILER_STARTUP_INTERVAL: '2', MOZ_PROFILER_STARTUP_ENTRIES: '20000000', MOZ_PROFILER_STARTUP_FILTERS: '*' } : process.env });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
for (let i = 0; i < 100; i++) {
  const ok = await new Promise((r) => { const c = connect(port, '127.0.0.1'); c.on('connect', () => { c.destroy(); r(true); }); c.on('error', () => r(false)); });
  if (ok) break;
  await sleep(300);
}
const ws = new WebSocket(`ws://127.0.0.1:${port}/session`);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const pending = new Map();
ws.addEventListener('message', (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('session.new', { capabilities: {} });
const tree = await send('browsingContext.getTree', {});
const context = tree.result.contexts[0].context;
const evalJs = async (expr) => {
  const r = await send('script.evaluate', { expression: expr, target: { context }, awaitPromise: true, resultOwnership: 'none' });
  if (r.result?.type === 'exception') return { exception: r.result.exceptionDetails?.text };
  return r.result?.result;
};
await send('browsingContext.navigate', { context, url, wait: 'complete' });
await sleep(1000);
await evalJs(`localStorage.setItem('mb-tutorial','done'); location.reload(); 1`);
await sleep(2500);
if (file) {
  const body = readFileSync(file, 'utf8');
  const r = await evalJs(`(async () => { ${body} })()`);
  console.log(JSON.stringify(r, null, 1));
}
if (process.env.JS) { const r = await evalJs(process.env.JS); console.error('pre', JSON.stringify(r).slice(0, 120)); await sleep(1500); }
// the CPU of every firefox.exe whose ancestry reaches our launch (the launcher re-executes
// itself, so the pid we spawned may already be gone: the profile path marks the main process)
const ps = (script) => execFileSync('powershell', ['-NoProfile', '-Command', script], { encoding: 'utf8' }).trim();
const cpuMs = () => +ps(`
$all = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'firefox.exe' }
$mine = @{}
foreach ($p in $all) { if ($p.CommandLine -like '*${dir.replace(/\\/g, '\\\\')}*' -or $p.ProcessId -eq ${ff.pid}) { $mine[$p.ProcessId] = 1 } }
$grew = $true
while ($grew) { $grew = $false; foreach ($p in $all) { if (-not $mine.ContainsKey($p.ProcessId) -and $mine.ContainsKey($p.ParentProcessId)) { $mine[$p.ProcessId] = 1; $grew = $true } } }
$t = 0.0
foreach ($q in $mine.Keys) { try { $t += (Get-Process -Id $q -ErrorAction Stop).TotalProcessorTime.TotalMilliseconds } catch {} }
"$t;$($mine.Count)"`).split(';')[0];
if (secs) {
  const t0 = cpuMs();
  const w0 = Date.now();
  await sleep(secs * 1000);
  const t1 = cpuMs();
  const wall = (Date.now() - w0) / 1000;
  console.log(`cpu ${((t1 - t0) / (wall * 1000) * 100).toFixed(1)}% of one core over ${wall.toFixed(1)} s`);
}
if (shotAt > 0) {
  const r = await send('browsingContext.captureScreenshot', { context });
  writeFileSync(process.argv[shotAt + 1], Buffer.from(r.result.data, 'base64'));
}
// a clean close makes the profiler write its file (a kill would lose it)
if (process.env.PROF) { await send('browser.close', {}); for (let i = 0; i < 60 && !existsSync(process.env.PROF); i++) await sleep(500); await sleep(1500); }
ws.close();
ps(`Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'firefox.exe' -and $_.CommandLine -like '*${dir.replace(/\\/g, '\\\\')}*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }`);
try { ff.kill(); } catch {}
process.exit(0);
