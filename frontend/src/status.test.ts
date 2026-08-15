import { describe, expect, it } from "vitest";
import { displayClassName, statusClass } from "./status";

describe("dashboard status helpers", () => {
  it("renders meaningful class and resolved status labels", () => {
    expect(displayClassName("cleanable_surface_issue")).toBe("cleanable surface issue");
    expect(statusClass("RESOLVED")).toBe("status-resolved");
  });
});
