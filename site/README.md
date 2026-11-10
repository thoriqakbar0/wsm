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
