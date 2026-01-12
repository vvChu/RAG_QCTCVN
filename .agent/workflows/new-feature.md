---
description: Create a new feature branch with proper naming and setup
---

# New Feature Workflow

This workflow creates a feature branch following project conventions.

## Steps

1. Ensure you're on the latest master:

   ```powershell
   git checkout master
   git pull origin master
   ```

2. Create the feature branch with descriptive name:

   ```powershell
   git checkout -b feature/<feature-name>
   ```

   Naming conventions:
   - `feature/<name>` - New functionality
   - `fix/<name>` - Bug fixes
   - `docs/<name>` - Documentation updates
   - `refactor/<name>` - Code refactoring

3. Make your changes and commit:

   ```powershell
   git add -A
   git commit -m "feat(<scope>): description"
   ```

4. Push the branch:
   // turbo

   ```powershell
   git push -u origin HEAD
   ```

5. Create a PR:
   // turbo

   ```powershell
   gh pr create --fill --web
   ```

## Commit Message Format

```
<type>(<scope>): <description>

Types: feat, fix, docs, refactor, test, chore
```
