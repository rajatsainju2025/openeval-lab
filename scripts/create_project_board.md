# Create GitHub Project Board (manual steps)

Use GitHub Projects (Beta) at https://github.com/rajatsainju2025/openeval-lab/projects

1) Click "New project" → Board.
2) Name: OpenEval Lab – 10-Day Plan.
3) Views: Board (Default), Table (Issues), Roadmap (Milestones).
4) Columns: Backlog, Planned (Day 1..10), In Progress, In Review, Done.
5) Fields: Status, Priority, Day, Size, Labels.
6) Automation: Auto-add issues/PRs with label `plan-10d`.
7) Save project link in README.

Optional: Create issues per task
- For each item in docs/10-day-contribution-plan.md, create an Issue with label `plan-10d` and Day=N.
- Assign yourself and add to the project board.

CLI option (gh):
- gh auth login
- gh project create --owner rajatsainju2025 --title "OpenEval Lab – 10-Day Plan" --format board
- gh project field-create --owner rajatsainju2025 --project "OpenEval Lab – 10-Day Plan" --name Day --data-type NUMBER
- gh issue create --title "Day 1: concurrency retries" --body "See docs/10-day-contribution-plan.md" --label plan-10d
- gh project item-add --owner rajatsainju2025 --project "OpenEval Lab – 10-Day Plan" --url https://github.com/rajatsainju2025/openeval-lab/issues/<ID>
