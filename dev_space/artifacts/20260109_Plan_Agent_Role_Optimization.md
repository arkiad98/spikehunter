# Agent Role Structure Optimization Plan

## Goal Description
Optimize the agent's role persistence and instruction loading by converting the single `.rules.md` file into a tiered structure of standard markdown files (`.md`) and enforcing their review via a workflow. This addresses the issue of the agent "forgetting" its role or failing to parse `.rules` extension files effectively.

## User Review Required
> [!NOTE]
> The existing `.agent/rules/.rules.md` file will be deleted after the new files are created to prevent conflict.

## Proposed Changes

### Agent Rules (`.agent/rules/`)

#### [NEW] [01_persona.md](file:///d:/spikehunter/.agent/rules/01_persona.md)
- Contains the Agent Persona (Quant Dev, Pro Trader, Macro Analyst).

#### [NEW] [02_principles.md](file:///d:/spikehunter/.agent/rules/02_principles.md)
- Contains Core Principles (Language Policy, Documentation) and Version Control rules.

#### [NEW] [03_engineering.md](file:///d:/spikehunter/.agent/rules/03_engineering.md)
- Contains Financial Integrity and Code Quality standards.

#### [DELETE] [.rules.md](file:///d:/spikehunter/.agent/rules/.rules.md)
- Remove the obsolete monolithic file.

### Agent Workflows (`.agent/workflows/`)

#### [NEW] [daily_task.md](file:///d:/spikehunter/.agent/workflows/daily_task.md)
- A workflow file instructing the agent to review the `01_persona.md` and `02_principles.md` files at the start of tasks.

## Verification Plan

### Automated Verification
- **List Files**: Run `ls .agent/rules/` to confirm the presence of the new files and absence of the old one.
- **Content Check**: Use `cat` (or `view_file`) to verify the content of the new files matches the source material.

### Manual Verification
- The user can verify by checking the "Agent Settings" or "Active Rules" (if available in their UI) or simply observing if the agent adheres to the persona in subsequent turns.
