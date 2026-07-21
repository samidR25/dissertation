"""
apply_lopo_scatter_label_fix_patch.py
======================================
Fixes label collisions in generate_lopo_figures.py's Figure 2 (sensitivity
vs. FP/hr scatter). Several LOPO folds sit close together at low FP/hr and
at sensitivity==1.0 (chb05/chb18, chb13/chb07, chb06/chb03, chb11/chb10);
the original fixed-offset ax.annotate() renders these as unreadable merged
strings (observed in review: "0518", "187", "1110", "0603").

Fix: replace fixed-offset annotate() calls with adjustText's adjust_text(),
which auto-repels overlapping labels and draws a thin leader line back to
the true point.

Requires: pip install adjustText --break-system-packages

Anchor-based, hard-refusal on mismatch (project convention) -- run from the
repo root:
    python3 apply_lopo_scatter_label_fix_patch.py
"""
import sys

TARGET = "generate_lopo_figures.py"

OLD = '''# ── Figure 2: sensitivity vs FP/hr scatter, coloured by collapse PASS/FAIL ──
fig, ax = plt.subplots(figsize=(6.5, 5))
for r in rows:
    if r['event_sensitivity'] is None or r['fp_per_hour'] is None:
        continue
    color = PASS_COLOR if r['collapse_pass'] else FAIL_COLOR
    marker = 'o' if r['collapse_pass'] else 'x'
    ax.scatter(r['fp_per_hour'], r['event_sensitivity'], c=color, marker=marker,
               s=90, edgecolors='black' if r['collapse_pass'] else None, linewidths=0.8, zorder=3)
    ax.annotate(r['patient'].replace('chb', ''), (r['fp_per_hour'], r['event_sensitivity']),
                textcoords="offset points", xytext=(6, 3), fontsize=8)
ax.set_xlabel('False positives / hour')'''

NEW = '''# ── Figure 2: sensitivity vs FP/hr scatter, coloured by collapse PASS/FAIL ──
fig, ax = plt.subplots(figsize=(6.5, 5))
texts = []
for r in rows:
    if r['event_sensitivity'] is None or r['fp_per_hour'] is None:
        continue
    color = PASS_COLOR if r['collapse_pass'] else FAIL_COLOR
    marker = 'o' if r['collapse_pass'] else 'x'
    ax.scatter(r['fp_per_hour'], r['event_sensitivity'], c=color, marker=marker,
               s=90, edgecolors='black' if r['collapse_pass'] else None, linewidths=0.8, zorder=3)
    texts.append(ax.text(r['fp_per_hour'], r['event_sensitivity'], r['patient'].replace('chb', ''),
                          fontsize=8, zorder=4))

# Several points sit close together at low FP/hr and at sensitivity==1.0
# (chb05/chb18, chb13/chb07, chb06/chb03, chb11/chb10) -- plain fixed-offset
# annotate() renders these as unreadable merged strings. adjustText nudges
# overlapping labels apart automatically and draws a thin leader line back
# to the true point so the association stays unambiguous.
try:
    from adjustText import adjust_text
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.7),
                expand_points=(1.4, 1.6), force_points=(0.3, 0.4))
except ImportError:
    raise ImportError(
        "adjustText is required to keep close-together point labels readable "
        "(pip install adjustText --break-system-packages). Falling back to "
        "overlapping fixed-offset labels would reintroduce the collision bug "
        "this patch fixes.")
ax.set_xlabel('False positives / hour')'''


def main():
    with open(TARGET, "r") as f:
        content = f.read()

    if OLD not in content:
        print(f"REFUSED: anchor text not found in {TARGET}. "
              f"File may already be patched, or has diverged from the "
              f"expected version. No changes made.")
        sys.exit(1)

    if content.count(OLD) > 1:
        print(f"REFUSED: anchor text is not unique in {TARGET} "
              f"({content.count(OLD)} occurrences). No changes made.")
        sys.exit(1)

    content = content.replace(OLD, NEW)
    with open(TARGET, "w") as f:
        f.write(content)

    print(f"Patched {TARGET}: Figure 2 scatter labels now use adjustText "
          f"for collision-free placement.")


if __name__ == "__main__":
    main()
