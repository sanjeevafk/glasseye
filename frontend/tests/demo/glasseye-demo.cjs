'use strict';
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const BASE_URL = process.env.QA_BASE_URL || 'http://127.0.0.1:8000';
const VIDEO_DIR = path.join(__dirname, 'recordings');
const OUTPUT_NAME = 'glasseye-demo.webm';
const REHEARSAL = process.argv.includes('--rehearse');

async function injectCursor(page) {
  await page.evaluate(() => {
    if (document.getElementById('demo-cursor')) return;
    const cursor = document.createElement('div');
    cursor.id = 'demo-cursor';
    cursor.innerHTML = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5 3L19 12L12 13L9 20L5 3Z" fill="white" stroke="black" stroke-width="1.5" stroke-linejoin="round"/>
    </svg>`;
    cursor.style.cssText = `
      position: fixed; z-index: 999999; pointer-events: none;
      width: 24px; height: 24px;
      transition: left 0.1s, top 0.1s;
      filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.3));
    `;
    cursor.style.left = '0px';
    cursor.style.top = '0px';
    document.body.appendChild(cursor);
    document.addEventListener('mousemove', (e) => {
      cursor.style.left = e.clientX + 'px';
      cursor.style.top = e.clientY + 'px';
    });
  });
}

async function injectSubtitleBar(page) {
  await page.evaluate(() => {
    if (document.getElementById('demo-subtitle')) return;
    const bar = document.createElement('div');
    bar.id = 'demo-subtitle';
    bar.style.cssText = `
      position: fixed; bottom: 0; left: 0; right: 0; z-index: 999998;
      text-align: center; padding: 12px 24px;
      background: rgba(0, 0, 0, 0.75);
      color: white; font-family: -apple-system, "Segoe UI", sans-serif;
      font-size: 16px; font-weight: 500; letter-spacing: 0.3px;
      transition: opacity 0.3s;
      pointer-events: none;
    `;
    bar.textContent = '';
    bar.style.opacity = '0';
    document.body.appendChild(bar);
  });
}

async function showSubtitle(page, text) {
  await page.evaluate((t) => {
    const bar = document.getElementById('demo-subtitle');
    if (!bar) return;
    if (t) {
      bar.textContent = t;
      bar.style.opacity = '1';
    } else {
      bar.style.opacity = '0';
    }
  }, text);
  if (text) await page.waitForTimeout(800);
}

async function ensureVisible(page, locator, label) {
  const el = typeof locator === 'string' ? page.locator(locator).first() : locator;
  const visible = await el.isVisible().catch(() => false);
  if (!visible) {
    const msg = `REHEARSAL FAIL: "${label}" not found`;
    console.error(msg);
    const found = await page.evaluate(() => {
      return Array.from(document.querySelectorAll('button, input, select, textarea, a, [data-testid]'))
        .filter(el => el.offsetParent !== null)
        .map(el => `${el.tagName} "${el.textContent?.trim().substring(0, 30)}" testid="${el.getAttribute('data-testid') || ''}"`)
        .join('\n  ');
    });
    console.error('  Visible elements:\n  ' + found);
    return false;
  }
  console.log(`REHEARSAL OK: "${label}"`);
  return true;
}

async function moveAndClick(page, locator, label, opts = {}) {
  const { postClickDelay = 800, ...clickOpts } = opts;
  const el = typeof locator === 'string' ? page.locator(locator).first() : locator;
  const visible = await el.isVisible().catch(() => false);
  if (!visible) {
    console.error(`WARNING: moveAndClick skipped - "${label}" not visible`);
    return false;
  }
  try {
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(300);
    const box = await el.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 10 });
      await page.waitForTimeout(400);
    }
    await el.click(clickOpts);
  } catch (e) {
    console.error(`WARNING: moveAndClick failed on "${label}": ${e.message}`);
    return false;
  }
  await page.waitForTimeout(postClickDelay);
  return true;
}

async function panElements(page, selector, maxCount = 6) {
  const elements = await page.locator(selector).all();
  for (let i = 0; i < Math.min(elements.length, maxCount); i++) {
    try {
      const box = await elements[i].boundingBox();
      if (box && box.y < 700) {
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 8 });
        await page.waitForTimeout(600);
      }
    } catch (e) {
      console.warn(`WARNING: panElements skipped element ${i}: ${e.message}`);
    }
  }
}

(async () => {
  const browser = await chromium.launch({ headless: true });

  if (REHEARSAL) {
    const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
    const page = await context.newPage();
    await page.goto(BASE_URL);
    await page.waitForTimeout(2000);
    const steps = [
      { label: 'Facade canvas', selector: '[data-testid="facade-canvas"]' },
      { label: 'Run demo button', selector: '[data-testid="run-demo"]' },
      { label: 'Timeline', selector: '[data-testid="timeline"]' },
      { label: 'Cleanable issue card', selector: '[data-testid="issue-cleanable_surface_issue"]' },
      { label: 'Structural issue card', selector: '[data-testid="issue-structural_issue"]' },
    ];
    let allOk = true;
    for (const step of steps) {
      if (!(await ensureVisible(page, step.selector, step.label))) allOk = false;
    }
    await browser.close();
    process.exit(allOk ? 0 : 1);
  }

  const context = await browser.newContext({
    recordVideo: { dir: VIDEO_DIR, size: { width: 1280, height: 720 } },
    viewport: { width: 1280, height: 720 }
  });
  const page = await context.newPage();

  try {
    await page.goto(BASE_URL);
    await injectCursor(page);
    await injectSubtitleBar(page);
    await page.waitForTimeout(2500);

    await showSubtitle(page, 'GlassEye - facade inspection mission control');
    await panElements(page, '[data-testid^="panel-"]', 5);
    await panElements(page, '[data-testid="issue-"]', 2);
    await page.waitForTimeout(1500);
    await showSubtitle(page, '');

    await showSubtitle(page, 'Running the deterministic inspection mission');
    await moveAndClick(page, '[data-testid="run-demo"]', 'Run demo', { postClickDelay: 2500 });
    await showSubtitle(page, 'YOLO detects a cleanable and a structural issue');

    const actuatorModal = page.getByTestId('actuator-command-modal');
    await actuatorModal.waitFor({ state: 'visible', timeout: 180_000 });
    await page.waitForTimeout(1500);
    await showSubtitle(page, 'Cleaning command dispatched (simulated only)');
    await page.waitForTimeout(2000);
    await moveAndClick(page, actuatorModal.getByRole('button', { name: 'ACKNOWLEDGE' }), 'Acknowledge cleaning');
    await showSubtitle(page, '');

    const maintenanceModal = page.getByTestId('maintenance-dispatch-modal');
    await maintenanceModal.waitFor({ state: 'visible', timeout: 30_000 });
    await page.waitForTimeout(1500);
    await showSubtitle(page, 'Structural issue - maintenance dispatch, no cleaning');
    await page.waitForTimeout(2000);
    await moveAndClick(page, maintenanceModal.getByRole('button', { name: 'ACKNOWLEDGE' }), 'Acknowledge maintenance');
    await showSubtitle(page, '');

    const cleanable = page.getByTestId('issue-cleanable_surface_issue');
    const structural = page.getByTestId('issue-structural_issue');
    await cleanable.waitFor({ state: 'visible', timeout: 180_000 });
    await cleanable.locator('.status').filter({ hasText: 'RESOLVED' }).first().waitFor({ timeout: 180_000 });
    await structural.locator('.status').filter({ hasText: 'ESCALATED' }).first().waitFor({ timeout: 180_000 });
    await page.waitForTimeout(2000);

    await showSubtitle(page, 'Cleanable issue: cleaned and verified RESOLVED');
    await page.waitForTimeout(2500);
    await showSubtitle(page, 'Structural issue: ESCALATED for human review');
    await panElements(page, '[data-testid="vlm-review-structural_issue"]', 1);
    await page.waitForTimeout(2500);
    await showSubtitle(page, '');

    await showSubtitle(page, 'Full event timeline - 28 events, replayable');
    await panElements(page, '[data-testid="timeline"] li', 10);
    await page.waitForTimeout(2000);
    await showSubtitle(page, '');

    await page.waitForTimeout(2000);
    await page.screenshot({ path: path.join(VIDEO_DIR, 'glasseye-dashboard-final.png'), fullPage: true });
  } catch (err) {
    console.error('DEMO ERROR:', err.message);
  } finally {
    await context.close();
    const video = page.video();
    if (video) {
      const src = await video.path();
      const dest = path.join(VIDEO_DIR, OUTPUT_NAME);
      try {
        fs.copyFileSync(src, dest);
        console.log('Video saved:', dest);
      } catch (e) {
        console.error('ERROR: Failed to copy video:', e.message);
      }
    }
    await browser.close();
  }
})();
