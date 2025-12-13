---
description: Conduct a comprehensive review of the current codebase against the Product Requirements Document (PRD) to ensure functional compliance, and verify the accuracy of the README documentation.
---

**Instructions for the Agent:**

1.  **Load and Analyze:**
    * Load and parse the full content of the local file **`PRD.md`**.
    * Reference and deeply understand the specific **functional requirements** and **expected public behavior** outlined in the **Product Requirements Document (PRD)**.

2.  **Perform Verification Check:**
    * **Core Functionality Check:** Systematically check the source code against the PRD. Your primary task is to confirm that the **main functional output and observable behavior** of the application are exactly as specified in the PRD.
    * **Implementation Tolerance:** **IGNORE** variations in internal code structure, specific algorithms used, variable names, or the overall file structure. The code is allowed to vary, but the **main functionality MUST be identical** to the PRD's specification.


3.  **Report Generation:**
    * Produce a clear, detailed, and structured report.
    * The report must begin with a pass/fail statement regarding the **main functional compliance** with the PRD.
    * List all found discrepancies under two separate headings: "Functional/Behavioral Non-compliance" and "README Documentation Errors."