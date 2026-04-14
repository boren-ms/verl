# Workspace Instructions

## Task Planning and Completion

Every request MUST follow this workflow:

1. **Plan first**: Break the request into concrete, actionable steps using the todo list. Each step should be specific enough to verify completion (e.g., "Edit X to add Y" not "Update code").

2. **Track progress**: Mark each step in-progress before starting it, and mark it completed immediately after finishing. Only one step should be in-progress at a time.

3. **Do not stop early**: Continue working through all planned steps until the entire task is done. If blocked, diagnose and try alternative approaches rather than stopping and reporting the blocker.

4. **Verify completion**: After all steps are done, validate the result:
   - For code changes: check for errors, run relevant tests, or confirm the edit is syntactically correct.
   - For shell commands: confirm the command succeeded (check exit codes, inspect output).
   - For file operations: verify the file exists and contains the expected content.
   - For multi-file changes: ensure consistency across all modified files.

5. **Never leave tasks unfinished**: If the user's request implies multiple sub-tasks (numbered items, comma-separated requests, or compound instructions), address every single one. Do not selectively skip items.
