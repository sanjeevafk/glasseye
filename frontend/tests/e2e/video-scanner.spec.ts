import { expect, test } from "@playwright/test";

test("aerial drone video scanner loads presets, executes flight inference, and displays 4x3 damage heatmap", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("#drone-video-scanner")).toBeVisible();

  // Verify presets exist
  const analyzeBtn = page.getByTestId("analyze-video-btn");
  await expect(analyzeBtn).toBeVisible();

  // Click on the pre-inspection flight preset
  const presetBtn = page.getByTestId("preset-video-drone_flight_preinspection");
  if (await presetBtn.isVisible()) {
    await presetBtn.click();
  }

  // Click ANALYZE DRONE FLIGHT
  await analyzeBtn.click();

  // Verify loading state appears and then completes
  const videoPlayer = page.getByTestId("main-video-player");
  await expect(videoPlayer).toBeVisible({ timeout: 45_000 });

  // Verify 4x3 Panel Damage Heatmap renders
  const heatmap = page.getByTestId("panel-damage-heatmap");
  await expect(heatmap).toBeVisible({ timeout: 45_000 });
  await expect(heatmap.locator(".heatmap-cell")).toHaveCount(12);

  // Verify timeline scrubber is visible
  const scrubber = page.getByTestId("video-timeline-scrubber");
  await expect(scrubber).toBeVisible();

  // Verify Highlights reel is rendered if detections exist
  const highlights = page.getByTestId("video-highlights-reel");
  if (await highlights.isVisible()) {
    await expect(highlights.locator(".highlight-card")).not.toHaveCount(0);
  }
});
