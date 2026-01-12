---
description: Discard an experimental branch
---

# Discard Feature Workflow

Safely remove an experimental or abandoned branch.

## Steps

1. Switch to master:
   // turbo

   ```powershell
   git checkout master
   ```

2. Delete local branch:

   ```powershell
   git branch -D <branch-name>
   ```

3. Delete remote branch (if pushed):

   ```powershell
   git push origin --delete <branch-name>
   ```

4. Close any open PR:

   ```powershell
   gh pr close <pr-number> --comment "Discarding this feature"
   ```
