---
trigger: always_on
---

* Context: We are building an AI agent for trading.
* **Critical Instruction:** Before creating a plan, the agent must always read the contents of the file **PRD.md**.
* **Approval Requirement:** The agent must create the plan and explicitly seek user approval before proceeding with execution.
* **Constraint:** Do not perform any actions or tasks that are not explicitly mentioned in the **PRD.md* file.