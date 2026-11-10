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
      title: 'Introduction to Weight Space Mods',
      href: '/',
      text: "Why WSM? What is weight space programming? How do we do it? Why this design? What's done? Python programs transform named model tensors. WeightModel and execute_program record baseline and candidate evaluations and save standalone checkpoints. Research context includes LoRA, ROME, and task arithmetic. Released on PyPI as wsm 0.1.0. Recorded Modal and Spindle demonstrations have limited integration evidence, not general capability claims.",
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
