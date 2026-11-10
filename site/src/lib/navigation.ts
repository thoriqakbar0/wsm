export const repository = 'https://github.com/thoriqakbar0/wsm';
export const navigation = [
  { title: 'Get started', pages: [
    { title: 'Introduction', href: '/' },
    { title: 'Quickstart', href: '/docs/quickstart/' },
    { title: 'Python editing', href: '/docs/coding/' },
  ] },
  { title: 'Reference', pages: [
    { title: 'Architecture', href: '/docs/architecture/' },
    { title: 'CLI coming soon', href: '/docs/cli/' },
    { title: 'Research limitations', href: '/docs/research/' },
  ] },
];
export const pages = navigation.flatMap((group) => group.pages);
