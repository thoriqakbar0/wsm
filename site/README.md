# WSM documentation site

The documentation site uses Astro and generates static HTML. Run these commands from `site/` with Node.js 22.23.4 or later.

```sh
npm ci
npm run dev
```

Open `http://localhost:4321`. To check and build the site:

```sh
npm run check
npm run build
npm run preview
```

Deploy the contents of `site/dist/` to a static host at its domain root. No server adapter, React runtime, or credentials are required.

## Content

`src/content.config.ts` reads `../docs/**/*.md` directly. Edit those source documents to update the generated article pages. `src/lib/navigation.ts` defines the sidebar order. The introduction and offline quickstart are Astro pages under `src/pages/`.

The Markdown processor converts links between repository docs to local routes. Links to other repository files point to GitHub. Search indexes all documentation at build time and loads its JSON only when opened. Search matches words in titles and article text without an external service.

## Browser checks

```sh
npx playwright install chromium
npm test
```

Tests build the static site, start the local preview, and exercise search, navigation, copy, themes, article links, and responsive layouts. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to use an existing Chromium installation. Screenshots remain in the ignored `test-results/` directory.

## Visual identity

Weight Space Mods uses WSM as shorthand. The bracketed matrix mark represents direct access to named tensors, with one cell selected for modification. Use `public/brand-mark.svg` for the logo and `public/favicon.svg` for the browser icon. Keep clear space of at least one cell around the mark. The SVG uses large, aligned shapes so the brackets and selected cell remain legible at 16px.

The default palette is graphite `#101512`, panel `#141b17`, phosphor `#91e6aa`, foreground `#e5eee8`, supporting text `#a3b5a9`, and border `#688573`. Green identifies edits, links, focus, and primary actions. A saved light-theme preference remains available, with a darker green accent for contrast. Monospace headings and controls pair with Manrope body text; code and tensor values use monospace. Avoid scanlines, simulated logs, and decorative motion. Tensor values on the landing page are illustrative, not experimental results.

The landing page follows the SDK contract: evaluate the unchanged model, edit named tensors, evaluate on the same cases, and explicitly export a checkpoint. The runner records comparisons without automatically choosing a winner. The public brand changes neither the `wsm` import nor the CLI command.

## Production deployment

The static site is served by the `wsm-site` Cloudflare Worker at <https://wsm.ta-0.com>. `cloudflare.config.ts` owns the custom domain and static HTML/404 behavior. Cloudflare provisions DNS and TLS for the domain. No server code or Astro adapter is deployed.

From `site/`, run `npm run deploy` with Cloudflare CLI authentication to build and publish. To validate without publishing, run `npm run build:deploy` followed by `npm run deploy:check`. The packaging script copies only `dist/` into Cloudflare Build Output and validates it before deployment. The Cloudflare CLI and its Build Output helpers are pinned together because these APIs are in beta.

The `Site` GitHub Actions workflow checks pull requests and changes to `main`. It runs Astro checks, browser tests, and a deployment dry run. Only `main` deploys, using the exact verified artifact rather than rebuilding it. Manual dispatch on `main` also deploys. Production deployments are serialized and check the public homepage and quickstart afterward.

Configure repository secrets `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. Use an API token scoped to the deployment account with Workers Scripts Edit and Account Settings Read, plus Workers Routes Edit and Zone Read for `ta-0.com`. Keep credentials outside Git. The workflow never exposes deployment secrets to pull request jobs.

## Social previews

The shared layout emits canonical production URLs, Open Graph metadata, and Twitter large-image cards. All pages use the committed 1200 × 630 PNG at `public/og/wsm.png`, with page-specific titles and descriptions. `astro.config.mjs` defines the production origin so preview hosts and query parameters do not leak into shared URLs.

To regenerate the image, install the Playwright Chromium browser and run `npm run generate:og` from `site/`. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to use an existing Chromium installation. The generator uses the local logo and Manrope font without network requests. Inspect the PNG and commit it alongside any generator changes; production builds serve the committed image without requiring a browser.
