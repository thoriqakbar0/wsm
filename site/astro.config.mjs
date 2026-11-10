import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import { relative, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));

function repositoryLinks() {
  return (tree, file) => {
    function walk(node) {
      if (node.type === 'link' && !/^(?:[a-z]+:|#|\/)/i.test(node.url)) {
        const [path, fragment] = node.url.split('#');
        const source = relative(
          root,
          resolve(dirname(file.path), decodeURI(path)),
        );
        const anchor = fragment ? `#${fragment}` : '';
        node.url =
          source.startsWith('docs/') && source.endsWith('.md')
            ? `/docs/${source.slice(5, -3).toLowerCase()}/${anchor}`
            : `https://github.com/thoriqakbar0/wsm/blob/main/${source}${anchor}`;
      }
      node.children?.forEach(walk);
    }
    walk(tree);
  };
}

export default defineConfig({
  site: 'https://wsm.ta-0.com',
  output: 'static',
  markdown: {
    processor: unified({ remarkPlugins: [repositoryLinks] }),
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
  },
});
