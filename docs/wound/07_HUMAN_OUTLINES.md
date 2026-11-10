# 07. Final step: draw the wound outlines yourself, then run one command

> For: Ines. This is the last step of the wound-photo v2 work. Everything else (code, training, comparison,
> classifier update, tests) is automated; only the outlines need a human.
> Previous: [06_CLASSIFIER_V2.md](06_CLASSIFIER_V2.md). Background: [05_SEGMENTATION.md](05_SEGMENTATION.md), part 4.

## 1. Why this step matters

- To know whether the learned segmenter finds surgical cuts better than the classical one, both must be graded
  against **correct outlines** of the cuts ("ground truth").
- The 160 outlines used so far were drafted by an AI assistant. At human review (10 Nov 2026) they were judged
  **wrong: the lines did not follow the wounds**. So the incision results in 05 (Dice 0.46 vs 0.18) are **not
  reliable**, and the v2 segmenter was partly trained on wrong examples. They are kept only as history (journal V25).
- Your outlines fix both problems: the **40 test** outlines make the comparison trustworthy; the **120 training**
  outlines teach the model what a real incision looks like.
- What is NOT affected: the ulcer results (FUSeg, outlined by the dataset's experts) and the wound-attribute
  classifier results in 06 (graded on the surgeons' labels of SurgWound; the segmenter only places a crop box).

## 2. Before you start (2 minutes)

- The AI outlines were moved aside (`outline\test\labels_ai_draft`, `outline\train\labels_ai_draft`), so the tool
  starts from empty photos. Your outlines go into `outline\test\labels` and `outline\train\labels`.
- Close what you do not need (Docker Desktop, Java projects in VS Code): training later needs the memory.

## 3. Draw the outlines

Open the VS Code terminal (PowerShell) and paste:

```powershell
cd C:\Users\juba839\OneDrive\3eme\hackathons\hackcure_esen\HackCure
C:\Users\juba839\surgiguard_data\venv-train\Scripts\python d1\scripts\outline_wounds.py C:\Users\juba839\surgiguard_data\outline\test
```

A window opens with the first photo not yet outlined (if you do not see it, it is behind VS Code: check the taskbar).

### What to select

| Draw | Do not draw |
|---|---|
| the cut itself: its edges and its scab | the redness around it |
| an open wound: the wound bed (hole, ulcer, raw area) | stitch threads, staples, glue strips, dressings |
| every separate scar on the photo (several shapes) | pen / marker drawings, shadows, folds of skin |

### How to select

**Thin cut (stitches, staples, a line)** - default tool, key `l`:
1. Left-click points **along** the cut, from one end to the other, following its curve (more clicks on curves).
2. **Mouse wheel**: make the green band as wide as the cut, not wider.
3. **Enter** to finish this line.

**Open or round wound** - key `p`:
1. Click **around the edge** of the wound.
2. **Enter** to close the shape.

**Several scars on one photo**: draw the first, **Enter**, draw the next, **Enter** ... then save once.

**Mistakes**: right-click removes the last point; `r` restarts the photo.

### Save and move

| Key | Action |
|---|---|
| `s` | save this photo and show the next one |
| `e` | no wound visible on this photo (saves an empty outline) |
| `n` | skip for now (come back later with `b`) |
| `b` | previous photo |
| `q` | quit; next time the tool continues where you stopped |

### Order

1. **The 40 test photos first** (the command above, about 40 min). Required.
2. **The 120 training photos** when you can (same command, ending with `outline\train`, about 2 h, can be split over
   several days). Recommended: without them, the models are only compared, not retrained.

## 4. After selecting: one command does the rest

When the test set is complete (and, ideally, the training set too), run:

```powershell
cd C:\Users\juba839\OneDrive\3eme\hackathons\hackcure_esen\HackCure
powershell -ExecutionPolicy Bypass -File d1\scripts\after_outlines.ps1
```

What it does, in order (`-DryRun` shows the commands without running them):

| # | Step | Time | Output |
|---|---|---|---|
| 1 | Counts your outlines; stops if the test set is incomplete | instant | |
| 2 | If ≥ 20 training photos are outlined: fine-tunes the segmenter on them (GPU, resumes by itself after a crash) | ~30 min | `d1\models\wound_seg\unet_v3.onnx` + `unet_v3.json` |
| 3 | Grades classical, v1, v2 and v3 on **your** 40 test outlines | ~1 min | `d1\models\wound_seg\compare_incisions_human.json` + a table on screen |
| 4 | If v3 exists: retrains the wound-attribute classifier with crops from v3 (previous one backed up in `surgiguard_data\backup_classifier_<date>`) | ~15 min | `d1\models\wound_heads.json` |
| 5 | Runs the tests | ~1 min | must say "passed" |

### How to read the table on screen

- **dice_mean / dice_median**: overlap with your outlines (0 = none, 1 = perfect). Higher is better.
- **width_ratio_median**: drawn width / true width; 1.0 is perfect, 2.0 = twice too wide (the classical weakness).
- **misses**: photos where the segmenter found nothing although you drew a wound. Lower is better.

## 5. Then (finishing the project)

Either tell Claude **"outlines done"** (it will write the results into the docs, the journal and commit), or do it
yourself:

1. Copy the table from step 3 into [05_SEGMENTATION.md](05_SEGMENTATION.md) ("Results" section), replacing the
   AI-outline numbers, and say which segmenter wins on your outlines.
2. Decide the default segmenter: if v3 (or v2) clearly beats classical on your test outlines, it can become the default
   in `d1\src\surgiguard_d1\wound\pipeline.py` (`WoundAnalyzer(segmenter=...)`); otherwise keep classical and write why.
3. Add a journal entry (V26) in [02_PROBLEMS_AND_SOLUTIONS.md](02_PROBLEMS_AND_SOLUTIONS.md): what changed with
   human outlines.
4. Commit and push (from `HackCure`):
   ```powershell
   git add d1\models d1\scripts docs
   git commit -m "Wound CV v2: human outlines, segmenter v3, comparison on human test outlines"
   git push
   ```
   The models are ~58 MB; if the push times out on a slow connection, run `git push` again.

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| No window appears | look behind VS Code in the taskbar; the terminal must show the help text |
| "paging file is too small" / MemoryError | close Docker Desktop and VS Code Java projects, run the command again (it resumes) |
| The script says the test set is incomplete | some photos were skipped with `n`: run the outline command again, it opens the missing ones |
| A photo was saved wrong | delete its file in `outline\test\labels` (or `train\labels`) and run the outline command again |
| You want the AI drafts back | they are in `labels_ai_draft` next to `labels` |
