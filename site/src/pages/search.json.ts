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
      text: 'WSM. Inspect weights, propose edits, evaluate candidates, retain strict improvements. Replayable evidence and bounded experiments.',
    },
    {
      title: 'Quickstart: run your first experiment',
      href: '/docs/quickstart/',
      text: 'Set WSM_TEACHER_URL, WSM_TEACHER_MODEL, and WSM_TEACHER_API_KEY. Run wsm run on the bundled addition task. The model inspects output.weight and edits one coefficient.',
    },
  );
  return new Response(JSON.stringify(entries), {
    headers: { 'Content-Type': 'application/json' },
  });
};
