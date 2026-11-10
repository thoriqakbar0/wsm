import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const root = join(import.meta.dirname, '../..');
const files = [join(root, 'README.md')];

function walk(dir) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist') continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      walk(path);
      continue;
    }
    if (/\.(md|astro|ts|js|mjs)$/.test(name)) files.push(path);
  }
}

walk(join(root, 'docs'));
walk(join(root, 'site/src'));

const banned = [
  [/\.venv\/bin\/python/g, '.venv/bin/python'],
  [/\.venv\/bin\/wsm/g, '.venv/bin/wsm'],
  [/python3 -m venv/g, 'python3 -m venv'],
  [/source \.venv/g, 'source .venv'],
];

const hits = [];
for (const file of files) {
  const lines = readFileSync(file, 'utf8').split('\n');
  lines.forEach((line, index) => {
    for (const [pattern, label] of banned) {
      pattern.lastIndex = 0;
      if (!pattern.test(line)) continue;
      if (label === 'python3 -m venv' && file === join(root, 'README.md') && line === 'python3 -m venv .venv') continue;
      if (label === '.venv/bin/python' && line.includes('INTEGRATION_PYTHON')) continue;
      hits.push(`${file}:${index + 1}: ${label}`);
    }
  });
}

if (hits.length) {
  console.error(hits.join('\n'));
  process.exit(1);
}

console.log(`command spelling ok (${files.length} files)`);
