import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from '@playwright/test';

const root = new URL('../', import.meta.url);
const logo = await readFile(new URL('public/brand-mark.svg', root), 'utf8');
const font = await readFile(
  new URL(
    'node_modules/@fontsource-variable/manrope/files/manrope-latin-wght-normal.woff2',
    root,
  ),
);
const browser = await chromium.launch({
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
});
try {
  const page = await browser.newPage({
    viewport: { width: 1200, height: 630 },
    deviceScaleFactor: 1,
  });
  await page.setContent(`<!doctype html><html lang="en"><meta charset="utf-8"><style>
    @font-face { font-family: Manrope; src: url(data:font/woff2;base64,${font.toString('base64')}); font-weight: 200 800; }
    * { box-sizing: border-box; }
    body { margin:0; width:1200px; height:630px; background:#101512; color:#e5eee8; font-family:Manrope,sans-serif; padding:56px 64px; }
    header { display:flex; align-items:center; gap:16px; font-size:25px; font-weight:650; }
    header svg { width:44px; height:44px; }
    .brand { color:#91e6aa; }
    .content { display:grid; grid-template-columns:790px 1fr; align-items:center; margin-top:64px; gap:20px; }
    h1 { font:600 62px/1.18 monospace; letter-spacing:-3.7px; margin:0; }
    p { color:#a3b5a9; font-size:24px; line-height:1.6; margin:24px 0 0; }
    .matrix { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; padding:15px 12px; border-inline:3px solid #91e6aa; }
    .cell { height:38px; background:#203c2a; }
    .selected { background:#91e6aa; }
    footer { position:absolute; bottom:52px; left:64px; right:64px; border-top:1px solid #2d3e33; padding-top:22px; display:flex; align-items:center; justify-content:space-between; font:18px monospace; }
    .command { color:#91e6aa; } .domain { color:#a3b5a9; }
  </style><body>
    <header>${logo}<span class="brand">Weight Space Mods</span></header>
    <div class="content"><div><h1>Program the edit.<br>Measure the model.</h1><p>Weight space programming with Python.<br>Named tensors. Evaluated edits. Standalone checkpoints.</p></div><div class="matrix" aria-hidden="true">${Array.from({ length: 9 }, (_, i) => `<span class="cell ${i === 4 ? 'selected' : ''}"></span>`).join('')}</div></div>
    <footer><span class="command">$ pip install 'wsm[editing]'</span><span class="domain">wsm.ta-0.com</span></footer>
  </body></html>`);
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({
    path: fileURLToPath(new URL('public/og/wsm.png', root)),
  });
} finally {
  await browser.close();
}
