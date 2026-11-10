import { getCollection } from 'astro:content';
import type { APIRoute } from 'astro';

export const GET: APIRoute = async () => {
  const docs = await getCollection('docs');
  const entries = docs.map((entry) => ({
    title: entry.body?.match(/^# (.+)$/m)?.[1] || entry.id,
    href: `/docs/${entry.id}/`,
    text: (entry.body || '')
      .replace(/[#*`\[\]()>|]/g, '')
      .replace(/\s+/g, ' ')
      .trim(),
  }));
  entries.unshift(
    {
      title: 'Introduction to Weight Space Modifiers',
      href: '/',
      text: 'WSM. Edit named model tensors with WeightModel, evaluate changes, and save standalone checkpoints.',
    },
    {
      title: 'Quickstart: edit weights with Python',
      href: '/docs/quickstart/',
      text: 'Install the editing dependencies. Use WeightModel or wsm execute to edit, evaluate, and save a local model.',
    },
  );
  return new Response(JSON.stringify(entries), {
    headers: { 'Content-Type': 'application/json' },
  });
};
