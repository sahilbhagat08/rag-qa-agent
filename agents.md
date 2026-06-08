# Git Workflow

Before making any code changes:

1. Create a feature branch.

Naming:

feature/<short-description>
bugfix/<short-description>
refactor/<short-description>

After implementation:

1. Run tests.
2. Run lint.
3. Show git diff summary.
4. Ask for approval.
5. Create commit.
6. Never push without approval.

Commit format:

feat: add rag evaluation pipeline
fix: resolve kafka consumer leak
refactor: simplify langgraph state management
