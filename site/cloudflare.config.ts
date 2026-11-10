import { defineConfig } from 'cf/config';

export default defineConfig({
  worker: {
    name: 'wsm-site',
    compatibilityDate: '2026-11-04',
    domains: ['wsm.ta-0.com'],
    workersDev: false,
    assets: {
      htmlHandling: 'auto-trailing-slash',
      notFoundHandling: '404-page',
    },
  },
});
