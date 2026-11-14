# Documentation website

The Markdown files in `docs/` are built into a static site with MkDocs
Material. The documentation tool is separate from the VLAForge runtime
dependencies.

## Preview locally

From the repository root, install the documentation dependencies and start the
local preview:

```bash
python -m pip install -r docs/requirements.txt
mkdocs serve
```

MkDocs serves the site locally and rebuilds pages as they change. To build the
static site without starting the preview server, run:

```bash
mkdocs build --strict
```

The generated site is written to `site/`, which is excluded from Git.

## Publish with GitHub Pages

1. Push the repository to GitHub.
2. Open repository **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.
4. Push a documentation change to the default branch.

The Documentation workflow checks the site on pull requests and pushes. It
deploys after a successful build from the default branch. A manual run is also
available through **Actions → Documentation → Run workflow**. GitHub shows the
published address in **Settings → Pages** and in the successful deployment
summary.
