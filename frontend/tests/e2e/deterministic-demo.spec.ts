import { expect, test } from "@playwright/test";

test("trained YOLO mission reaches the Three.js dashboard and replay", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("facade-canvas")).toBeVisible();
  await page.getByTestId("run-demo").click();
  const actuatorModal = page.getByTestId("actuator-command-modal");
  await expect(actuatorModal).toBeVisible({ timeout: 150_000 });
  await expect(actuatorModal).toContainText("CLEANING COMMAND DISPATCHED");
  await expect(actuatorModal).toContainText("SIMULATED_COMPLETE");
  await actuatorModal.getByRole("button", { name: "ACKNOWLEDGE" }).click();
  await expect(actuatorModal).toBeHidden();

  const cleanable = page.getByTestId("issue-cleanable_surface_issue");
  const structural = page.getByTestId("issue-structural_issue");
  await expect(cleanable).toContainText("RESOLVED", { timeout: 150_000 });
  await expect(structural).toContainText("ESCALATED", { timeout: 150_000 });
  await expect(cleanable.locator("img")).toBeVisible();
  await expect(structural.locator("img")).toBeVisible();
  await expect(page.getByTestId("timeline").locator("li")).toHaveCount(26);
  await expect(page.getByTestId("facade-canvas").locator("canvas")).toBeVisible();
  await page.screenshot({ path: "test-results/glasseye-dashboard.png", fullPage: true });
});
