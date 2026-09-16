# Electrons influence on the reionization of the Universe

Research project by Lautaro Carvalho.

## Research approaches

Both approaches are stored together on `main`, keeping their original internal layouts:

- [`approaches/Projecto-IA_aplications-Zaldarriaga/`](approaches/Projecto-IA_aplications-Zaldarriaga/): the project from Materias, including its code, manuscripts, figures, papers, and course material.
- [`approaches/Paper_1/`](approaches/Paper_1/): the project from Doctorado-Trabajo, including its notebooks, manuscripts, figures, and literature collections.

The root folders below remain available for shared material. Work inside the repository copies for future commits; edits to the original Desktop folders do not automatically synchronize here.

The import preserves source files without rewriting them. Some Python scripts in `Paper_1` contain absolute `/home/byaku/` paths and may need path adjustments on another computer. Copy integrity was checked; analyses and LaTeX builds were not rerun. Temporary LaTeX output and Python bytecode/checkpoints are excluded from Git; PDFs, images, bibliography outputs (`.bbl`), and historical source backups are retained. See [the import record](notes/import-record.md) for exact exclusions.

## Organization

| Folder | Contents |
| --- | --- |
| `code/` | Scripts, notebooks, and instructions for running analyses |
| `figures/` | Images and plots used in the research |
| `manuscripts/` | LaTeX sources, associated files, and manuscript PDFs |
| `references/` | Reference papers and BibTeX bibliographies |
| `notes/` | Research notes, decisions, and meeting summaries |

Keep an existing LaTeX project together when importing it so relative paths still work.
PDFs and images are tracked; temporary LaTeX build files are ignored.

## Save work to GitHub

Run these commands in this repository's folder after saving your files:

```bash
git status
git diff
git add approaches code figures manuscripts references notes README.md .gitignore
git diff --cached --stat
git commit -m "Describe what changed"
git push
```

Review the listed files before committing. A commit is a local checkpoint; a successful push uploads it to GitHub. Changes that are uncommitted, ignored, or on unpushed branches are not backed up there. Confirm important uploads on the GitHub website. Keep an additional independent backup of research material.

## Try a different version

Start with saved and committed work, then create a branch:

```bash
git switch main
git switch -c alternative-analysis
```

Edit files, commit as above, and upload the new branch:

```bash
git push -u origin alternative-analysis
```

After committing your changes, switch between versions with:

```bash
git switch main
git switch alternative-analysis
```

Switching branches updates the files in this folder. Branches are parallel lines of work; commits retain the history within each branch. To incorporate a finished branch into main, open a pull request on GitHub and review its changes before merging.

## Mark a milestone

On the commit you want to label:

```bash
git tag -a submitted-v1 -m "First submitted version"
git push origin submitted-v1
```

Use a new tag name for each milestone.

## Adding existing research

Copy files into the appropriate folders and check `git status` before committing. Start with a small set. Avoid copying credentials or private access tokens. Large datasets may need separate storage or Git LFS; regular GitHub repositories block individual files larger than 100 MiB.

If a file does not appear in `git status`, check whether a rule excludes it:

```bash
git check-ignore -v path/to/file
```

## Recover on another computer

After installing Git and authenticating with GitHub:

```bash
git clone https://github.com/LautaroCarvalho/electrons-reionization.git
cd electrons-reionization
git branch -a
```

The clone includes the history and remote branches uploaded to GitHub. Use `git switch branch-name` to work on another branch.
