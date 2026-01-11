---
description: Create a PR for the current branch
---

# Create PR Workflow

Create a pull request for the current feature branch.

## Steps

1. Ensure all changes are committed:

   ```powershell
   git status
   git add -A
   git commit -m "your commit message"
   ```

2. Push the branch to origin:
   // turbo

   ```powershell
   git push -u origin HEAD
   ```

3. Create the PR:
   // turbo

   ```powershell
   gh pr create --fill --web
   ```

4. Wait for CI checks to pass on GitHub.

5. Request review if needed:

   ```powershell
   gh pr edit --add-reviewer <username>
   ```
