# GitHub upload guide

After extracting the release ZIP:

```bash
cd FAS
git init
git add .
git commit -m "Initial public release of FAS"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Before submission, replace the paper's code placeholder with the **actual reachable repository URL**, then verify it in a private/incognito browser window.

Recommended repository description:

> Official implementation of FAS (Functional Ancestry Simplex): API-only multi-parent functional ancestry decomposition for foundation models.

Recommended topics:

`model-provenance`, `model-lineage`, `foundation-models`, `black-box-auditing`, `active-learning`, `conformal-prediction`, `model-merging`
