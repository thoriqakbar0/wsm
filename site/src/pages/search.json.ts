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
      text: 'WSM is an AI harness for modding model weights. An agent writes Python edits, evaluates the changed model, and inspects experiment history. WeightModel and execute_program provide weight access and saved checkpoints. Experimental Tinker SDK integration through Modal Spindle trains a separate editing student from Codex teacher actions. Hosted Tinker training has not been verified. CLI coming soon. Research context includes LoRA, ROME, and task arithmetic.',
    },
    {
      title: 'Quickstart: edit weights with Python',
      href: '/docs/quickstart/',
      text: 'Use uv to install the editing dependencies. Use the Python SDK to edit, evaluate, and save a local model. CLI coming soon.',
    },
  );
  return new Response(JSON.stringify(entries), {
    headers: { 'Content-Type': 'application/json' },
  });
};
