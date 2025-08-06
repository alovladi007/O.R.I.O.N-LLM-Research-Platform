# Clean Up Repository Branches

To keep only the working `enhanced-platform` branch as the main branch, follow these steps:

## Option 1: Make enhanced-platform the new main branch (Recommended)

This will replace the current main branch with your enhanced-platform branch:

```bash
# Switch to enhanced-platform
git checkout enhanced-platform

# Force push enhanced-platform to main
git push origin enhanced-platform:main --force

# Delete the old enhanced-platform branch (since main now has the same content)
git push origin --delete enhanced-platform

# Delete the cursor branch
git push origin --delete cursor/build-and-enhance-orion-research-platform-a0e3

# Update your local repository
git branch -D main  # Delete local main
git checkout -b main  # Create new main from enhanced-platform
git branch -D enhanced-platform  # Delete local enhanced-platform
```

## Option 2: Keep enhanced-platform as the only branch

If you prefer to keep the branch name as `enhanced-platform`:

```bash
# Delete main branch
git push origin --delete main

# Delete cursor branch  
git push origin --delete cursor/build-and-enhance-orion-research-platform-a0e3

# Set enhanced-platform as the default branch on GitHub
# Go to: Settings → Branches → Change default branch to "enhanced-platform"
```

## Option 3: Merge enhanced-platform into main

If you want to preserve history:

```bash
# Switch to main
git checkout main
git pull origin main

# Merge enhanced-platform
git merge enhanced-platform --no-ff -m "Merge enhanced ORION platform"

# Push updated main
git push origin main

# Delete other branches
git push origin --delete enhanced-platform
git push origin --delete cursor/build-and-enhance-orion-research-platform-a0e3
```

## After Cleanup

Your repository will have only one working branch with all the enhanced ORION platform features:
- Complete modular architecture
- All advanced features implemented
- Web UI with Streamlit
- Docker deployment ready
- Comprehensive documentation

## Important Notes

- **Option 1** is recommended as it keeps the standard `main` branch name
- Make sure to update any CI/CD pipelines if they reference specific branches
- The `--force` flag will overwrite history, so make sure you're okay with that
- Consider downloading a backup of the repository before making these changes