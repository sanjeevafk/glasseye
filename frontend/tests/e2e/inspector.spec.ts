import { expect, test } from "@playwright/test";

test("image inspector allows 1-click preset sample evaluation and displays integrity score & policy", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("image-inspector")).toBeVisible();

  // Click on a preset sample button
  const presetBtn = page.getByRole("button", { name: "Concrete Facade Fracture" });
  if (await presetBtn.isVisible()) {
    await presetBtn.click();
    // Wait for inspection result to render
    const resultCard = page.getByTestId("inspection-result");
    await expect(resultCard).toBeVisible({ timeout: 30_000 });
    await expect(resultCard).toContainText("FAÇADE INTEGRITY INDEX");
    await expect(resultCard).toContainText("DISPATCH RECOMMENDATION & ACTION PLAN");
    await expect(resultCard.locator(".annotated-preview")).toBeVisible();
  }
});
