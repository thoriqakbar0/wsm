import { test, expect } from '@playwright/test';

test('link previews use production URLs and a loadable social image', async ({
  page,
  request,
}) => {
  for (const path of ['/', '/docs/coding/']) {
    await page.goto(path);
    const canonical = `https://wsm.ta-0.com${path}`;
    await expect(page.locator('link[rel="canonical"]')).toHaveAttribute(
      'href',
      canonical,
    );
    await expect(page.locator('meta[property="og:url"]')).toHaveAttribute(
      'content',
      canonical,
    );
    await expect(page.locator('meta[property="og:title"]')).toHaveAttribute(
      'content',
      /Weight Space Mods/,
    );
    await expect(page.locator('meta[name="twitter:card"]')).toHaveAttribute(
      'content',
      'summary_large_image',
    );
    for (const selector of [
      'meta[property="og:image"]',
      'meta[name="twitter:image"]',
    ]) {
      await expect(page.locator(selector)).toHaveAttribute(
        'content',
        'https://wsm.ta-0.com/og/wsm.png',
      );
    }
  }
  const response = await request.get('/og/wsm.png');
  expect(response.ok()).toBe(true);
  expect(response.headers()['content-type']).toContain('image/png');
  const png = await response.body();
  expect(png.subarray(0, 8)).toEqual(
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  );
  expect(png.readUInt32BE(16)).toBe(1200);
  expect(png.readUInt32BE(20)).toBe(630);
});

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    if (!localStorage.getItem('wsm-theme'))
      localStorage.setItem('wsm-theme', 'light');
  });
});

test('new visitors get the dark theme and can navigate by keyboard', async ({
  browser,
}) => {
  const context = await browser.newContext({ colorScheme: 'light' });
  const page = await context.newPage();
  await page.goto('http://127.0.0.1:4322/');
  await expect(
    page.getByRole('button', { name: 'Switch to light mode' }),
  ).toBeVisible();
  await page.keyboard.press('Tab');
  await expect(
    page.getByRole('link', { name: 'Skip to content' }),
  ).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('main')).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(
    page.getByRole('link', { name: 'Start building', exact: true }),
  ).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL('http://127.0.0.1:4322/docs/quickstart/');
  await context.close();
});

test('read a guide, copy a command, and retain the theme across pages', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await page
    .getByRole('main')
    .getByRole('link', { name: 'Start building', exact: true })
    .click();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(
    'Edit model weights with Python',
  );
  await page
    .getByRole('button', { name: 'Copy code', exact: true })
    .first()
    .click();
  await expect(page.getByRole('status').first()).toContainText('Code copied');
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toContain('wsm[editing]==0.1.0');
  expect(copied).toContain('pip install');
  expect(copied).not.toContain('.venv/bin/python');
  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  await page
    .getByRole('link', { name: 'Python editing', exact: true })
    .first()
    .click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(
    'Modify weights with Python',
  );
  await page.getByRole('button', { name: 'Switch to light mode' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(errors).toEqual([]);
});

test('search full article text, handle no results, and restore focus', async ({
  page,
}) => {
  await page.goto('/');
  const trigger = page.getByRole('button', { name: /Search docs/ });
  await trigger.click();
  const dialog = page.getByRole('dialog', { name: 'Search documentation' });
  const input = dialog.getByRole('searchbox');
  await input.fill('checkpoint');
  await expect(dialog.getByRole('link').first()).toBeVisible();
  await input.fill('zzq-no-such-topic-qqz');
  await expect(dialog.getByRole('status')).toContainText('No results');
  await page.keyboard.press('Escape');
  await expect(trigger).toBeFocused();
  await page.keyboard.press('Control+k');
  await input.fill('quickstart');
  await expect(dialog.getByRole('link', { name: /Quickstart/ })).toBeVisible();
  await input.press('Enter');
  await expect(page).toHaveURL('/docs/quickstart/');
});

test('all generated internal links and anchors resolve', async ({
  page,
  request,
}) => {
  const index = await (await request.get('/search.json')).json();
  const missing: string[] = [];
  const destinations = new Map<string, Set<string>>();
  for (const entry of index) {
    await page.goto(entry.href);
    await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);
    const links = await page
      .locator('a[href]')
      .evaluateAll((anchors) =>
        anchors.map((anchor) => (anchor as HTMLAnchorElement).href),
      );
    for (const href of links) {
      const url = new URL(href);
      if (url.origin !== 'http://127.0.0.1:4322') continue;
      const anchors = destinations.get(url.pathname) || new Set<string>();
      if (url.hash) anchors.add(decodeURIComponent(url.hash.slice(1)));
      destinations.set(url.pathname, anchors);
    }
  }
  for (const [path, anchors] of destinations) {
    const response = await page.goto(path);
    if (response?.status() !== 200) missing.push(path);
    for (const anchor of anchors) {
      if (
        !(await page.evaluate(
          (id) => Boolean(document.getElementById(id)),
          anchor,
        ))
      )
        missing.push(`${path}#${anchor}`);
    }
  }
  expect(missing).toEqual([]);
});

for (const width of [
  320, 375, 638, 640, 768, 898, 900, 1024, 1198, 1200, 1440, 1599, 1601,
]) {
  test(`layout at ${width}px`, async ({ browser }, testInfo) => {
    const context = await browser.newContext({
      viewport: {
        width,
        height:
          (
            { 320: 568, 375: 812, 768: 1024, 1024: 768, 1440: 900 } as Record<
              number,
              number
            >
          )[width] || 800,
      },
      hasTouch: width < 1024,
      colorScheme: 'light',
    });
    const page = await context.newPage();
    await page.goto('http://127.0.0.1:4322/');
    await page.evaluate(() => document.fonts.ready);
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: testInfo.outputPath(`home-${width}.png`),
      fullPage: true,
    });
    if (width < 640) {
      await page.getByRole('button', { name: 'Open navigation' }).click();
      await page
        .getByRole('dialog', { name: 'Documentation navigation' })
        .getByRole('link', { name: 'Quickstart', exact: true })
        .click();
      await expect(page).toHaveURL('http://127.0.0.1:4322/docs/quickstart/');
    } else {
      await page.goto('http://127.0.0.1:4322/docs/coding/');
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
    await context.close();
  });
}

test('short landscape, large text, dark mode, and reduced motion', async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 812, height: 375 });
  await page.goto('/');
  await page.getByRole('button', { name: /Search docs/ }).click();
  const dialog = page.getByRole('dialog');
  await dialog.getByRole('searchbox').fill('checkpoint');
  await expect(dialog.getByRole('link').first()).toBeVisible();
  const bounds = await dialog.boundingBox();
  expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(375);
  await page.screenshot({ path: testInfo.outputPath('landscape-search.png') });
  await page.keyboard.press('Escape');
  await page.setViewportSize({ width: 1024, height: 768 });
  await page.evaluate(() => (document.documentElement.style.fontSize = '200%'));
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath('large-text.png'),
    fullPage: true,
  });
  await page.evaluate(() => (document.documentElement.style.fontSize = ''));
  const codeToken = page.locator('pre span[style]').first();
  const lightTokenColor = await codeToken.evaluate(
    (element) => getComputedStyle(element).color,
  );
  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  expect(
    await codeToken.evaluate((element) => getComputedStyle(element).color),
  ).not.toBe(lightTokenColor);
  await page.screenshot({
    path: testInfo.outputPath('dark.png'),
    fullPage: true,
  });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  expect(
    await page
      .getByRole('button', { name: 'Switch to light mode' })
      .evaluate((element) => getComputedStyle(element).transitionDuration),
  ).toBe('0s');
});
