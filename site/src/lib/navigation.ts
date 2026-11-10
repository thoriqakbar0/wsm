export const repository = 'https://github.com/thoriqakbar0/wsm';

export const navigation = [
  {
    title: 'Get started',
    pages: [
      { title: 'Introduction', href: '/' },
      { title: 'Quickstart', href: '/docs/quickstart/' },
      { title: 'Prompt-driven edits', href: '/docs/wsm/' },
    ],
  },
  {
    title: 'Core concepts',
    pages: [
      { title: 'Architecture', href: '/docs/architecture/' },
      { title: 'Experiment contract', href: '/docs/contract/' },
      { title: 'Sessions & checkpoints', href: '/docs/client/' },
      { title: 'Sandbox boundaries', href: '/docs/sandbox/' },
    ],
  },
  {
    title: 'Guides',
    pages: [
      { title: 'Configure an experiment', href: '/docs/driver/' },
      { title: 'Finish an experiment', href: '/docs/agent/' },
      { title: 'Verify your results', href: '/docs/verification/' },
    ],
  },
  {
    title: 'Reference',
    pages: [
      { title: 'CLI commands', href: '/docs/cli/' },
      { title: 'Tinker integration', href: '/docs/tinker/' },
      { title: 'Research & limitations', href: '/docs/research/' },
    ],
  },
];

export const pages = navigation.flatMap((group) => group.pages);
