import { fileURLToPath } from 'node:url';
import { loadAndParseConfig } from '@cloudflare/config';
import {
  cleanBuildOutputDir,
  readBuildOutput,
  writeAssets,
  writeRootConfig,
  writeWorkerConfig,
} from '@cloudflare/build-output-utils';

const root = fileURLToPath(new URL('../', import.meta.url));
const context = { mode: 'production', isPreview: false };
const { result } = await loadAndParseConfig(
  `${root}/cloudflare.config.ts`,
  context,
);
if (!result.success) throw result.error;
const { worker, accountId, complianceRegion } = result.data;
if (!worker) throw new Error('The site requires a Worker configuration.');
await cleanBuildOutputDir(root);
await writeAssets({ root, sourceDirectory: `${root}/dist` });
await writeWorkerConfig({ root, config: worker });
await writeRootConfig(root, { accountId, complianceRegion }, context);
await readBuildOutput(root);
console.log('Packaged Astro output for Cloudflare Static Assets.');
