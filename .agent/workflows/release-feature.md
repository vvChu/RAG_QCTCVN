---
description: Merge a PR after CI passes and clean up
---

# Release Feature Workflow

Merge an approved PR and clean up local branches.

## Steps

1. Ensure PR is approved and CI passes:

   ```powershell
   gh pr status
   ```

2. Merge the PR:
   // turbo

   ```powershell
   gh pr merge --squash --delete-branch
   ```

3. Update local master:
   // turbo

   ```powershell
   git checkout master
   git pull origin master
   ```

4. Clean up old local branches:
   // turbo

   ```powershell
   git branch -d <feature-branch-name>
   ```

5. Verify merge:

   ```powershell
   git log --oneline -5
   ```
