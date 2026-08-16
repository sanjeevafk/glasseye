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

  const maintenanceModal = page.getByTestId("maintenance-dispatch-modal");
  await expect(maintenanceModal).toBeVisible({ timeout: 15_000 });
  await expect(maintenanceModal).toContainText("STRUCTURAL MAINTENANCE DISPATCH");
  await expect(maintenanceModal).toContainText("ESCALATE — NO CLEANING");
  await maintenanceModal.getByRole("button", { name: "ACKNOWLEDGE" }).click();
  await expect(maintenanceModal).toBeHidden();

  const cleanable = page.getByTestId("issue-cleanable_surface_issue");
  const structural = page.getByTestId("issue-structural_issue");
  await expect(cleanable).toContainText("RESOLVED", { timeout: 150_000 });
  await expect(structural).toContainText("ESCALATED", { timeout: 150_000 });
  await expect(cleanable.locator("img")).toBeVisible();
  await expect(structural.locator("img")).toBeVisible();
  await expect(page.getByTestId("vlm-review-structural_issue")).toContainText("ESCALATE");
  await expect(page.getByTestId("timeline")).toContainText("VLM REVIEW RESULT");
  await expect(page.getByTestId("timeline").locator("li")).toHaveCount(28);
  await expect(page.getByTestId("facade-canvas").locator("canvas")).toBeVisible();
  await page.screenshot({ path: "test-results/glasseye-dashboard.png", fullPage: true });
});
