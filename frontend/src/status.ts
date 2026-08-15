import type { IssueStatus } from "./types";

export function statusClass(status: IssueStatus): string {
  if (status === "RESOLVED") return "status-resolved";
  if (status === "ESCALATED" || status === "UNRESOLVED") return "status-escalated";
  if (status === "REVIEW") return "status-review";
  return "status-active";
}

export function displayClassName(value: string): string {
  return value.replaceAll("_", " ");
}
