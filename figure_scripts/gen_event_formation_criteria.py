import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

RED = '#C44E52'
GREEN = '#55A868'
BLUE = '#4C72B0'
GREY = '#888888'

plt.rcParams['font.size'] = 9.5


def draw_cells(ax, x0, y0, states, cell_w=1.0, cell_h=0.8):
    """Draw len(states) unit cells starting at (x0,y0). states: list of bool
    (True=positive/red, False=negative/white). Returns (x_start, x_end)."""
    for i, s in enumerate(states):
        x = x0 + i * cell_w
        color = RED if s else 'white'
        rect = Rectangle((x, y0), cell_w * 0.92, cell_h, facecolor=color,
                          edgecolor='black', linewidth=0.9, zorder=3)
        ax.add_patch(rect)
    return x0, x0 + len(states) * cell_w


def bracket(ax, x1, x2, y, label, color='black', lw=1.3, fontsize=8.6, label_dy=0.32):
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle=']-[', mutation_scale=9,
                         linewidth=lw, color=color)
    ax.add_patch(a)
    ax.text((x1 + x2) / 2, y + label_dy, label, ha='center', va='bottom',
            fontsize=fontsize, color=color, weight='bold')


def gap_block(ax, x0, y0, w, h, label):
    """Hatched block representing a compressed (not-to-scale) time gap."""
    rect = Rectangle((x0, y0), w, h, facecolor='white', edgecolor=GREY,
                      linewidth=0.9, hatch='////', zorder=2)
    ax.add_patch(rect)
    ax.text(x0 + w / 2, y0 - 0.30, label, ha='center', va='top',
            fontsize=8.0, color=GREY, style='italic')


def longest_run(states):
    """Return (start_idx, length) of the longest consecutive True run."""
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i, s in enumerate(states):
        if s:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0
    return best_start, best_len


def coverage_bar(ax, x0, y0, h, pct, threshold_pct=30, bar_w=6.0):
    """Vertical-fill horizontal bar showing % coverage, with a dashed
    threshold line. Returns nothing; draws in place."""
    outline = Rectangle((x0, y0), bar_w, h, facecolor='white', edgecolor='black',
                         linewidth=1.0, zorder=3)
    ax.add_patch(outline)
    fill_color = GREEN if pct >= threshold_pct else RED
    fill_w = bar_w * (pct / 100.0)
    fill = Rectangle((x0, y0), fill_w, h, facecolor=fill_color, edgecolor='none',
                      alpha=0.85, zorder=4)
    ax.add_patch(fill)
    thresh_x = x0 + bar_w * (threshold_pct / 100.0)
    ax.plot([thresh_x, thresh_x], [y0 - 0.08, y0 + h + 0.08], color='black',
             linestyle='--', linewidth=1.2, zorder=5)
    ax.text(x0 + bar_w / 2, y0 + h + 0.22, f'{pct:.0f}% coverage', ha='center',
            fontsize=8.2, weight='bold', color=fill_color)
    ax.text(thresh_x, y0 - 0.20, '30%', ha='center', va='top', fontsize=7.0, color='black')


def run_highlight(ax, x0, start, length, y0, cell_w, cell_h, threshold=3, label_below=True):
    """Draw a coloured border around the longest-run cells and label it."""
    color = GREEN if length >= threshold else RED
    hx = x0 + start * cell_w
    hw = length * cell_w * 0.92 if length > 0 else 0.3
    rect = Rectangle((hx - 0.06, y0 - 0.06), hw + 0.12, cell_h + 0.12,
                      fill=False, edgecolor=color, linewidth=2.6, zorder=6)
    ax.add_patch(rect)
    verdict = f'\u2265{threshold} \u2192 pass' if length >= threshold else f'<{threshold} \u2192 fail'
    label = f'longest run = {length}  ({verdict})'
    ty = y0 - 0.35 if label_below else y0 + cell_h + 0.25
    va = 'top' if label_below else 'bottom'
    ax.text(hx + hw / 2, ty, label, ha='center', va=va, fontsize=8.0,
            weight='bold', color=color)


# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(12.0, 15.5))
gs = fig.add_gridspec(3, 1, height_ratios=[3.0, 3.9, 2.3], hspace=0.6)

XLIM = (-0.5, 30.5)

# =========================================================================
# PANEL A -- Event Grouping
# =========================================================================
axA = fig.add_subplot(gs[0])
axA.set_title('A.  Event Grouping  (60 s gap-tolerance rule)', fontsize=12.5,
               weight='bold', loc='left', pad=14)

# --- A1: gap 40s < 60s tolerance -> MERGE into one event ---
y1 = 3.2
x0a, x1a = draw_cells(axA, 0.5, y1, [True] * 10)
gap_block(axA, x1a, y1, 4.5, 0.8, '40 s gap\n(negative windows)')
x2a0, x3a = draw_cells(axA, x1a + 4.5, y1, [True] * 10)

bracket(axA, x0a, x3a, y1 + 1.35, 'ONE EVENT  --  40 s gap < 60 s tolerance \u2192 merged',
        color=GREEN, label_dy=0.28)
axA.text(x0a, y1 - 0.55, 'cluster 1 (10 positive windows)', fontsize=7.6, color='black')
axA.text(x2a0, y1 - 0.55, 'cluster 2 (10 positive windows)', fontsize=7.6, color='black')

# --- A2: gap 90s > 60s tolerance -> stays TWO events ---
y2 = 0.2
xb0, xb1 = draw_cells(axA, 0.5, y2, [True] * 7)
gap_block(axA, xb1, y2, 4.5, 0.8, '90 s gap\n(negative windows)')
xb2s, xb3 = draw_cells(axA, xb1 + 4.5, y2, [True] * 7)

bracket(axA, xb0, xb1, y2 - 0.55, 'event 1', color=RED, label_dy=-0.32)
bracket(axA, xb2s, xb3, y2 - 0.55, 'event 2', color=RED, label_dy=-0.32)
axA.text((xb0 + xb3) / 2, y2 - 1.15,
         'TWO SEPARATE EVENTS  --  90 s gap > 60 s tolerance \u2192 not merged',
         ha='center', fontsize=9.2, color=RED, weight='bold')

axA.set_xlim(*XLIM)
axA.set_ylim(-2.2, 5.4)
axA.axis('off')
axA.add_patch(Rectangle((XLIM[0] + 0.15, -2.1), XLIM[1] - XLIM[0] - 0.3, 7.3,
                          fill=False, edgecolor='black', linewidth=1.4))

# =========================================================================
# PANEL B -- Detection Criterion
# =========================================================================
axB = fig.add_subplot(gs[1])
axB.set_title('B.  Detection Criterion  (\u226530% window coverage AND \u22653 consecutive positive windows)',
               fontsize=12.0, weight='bold', loc='left', pad=14)

cell_w, cell_h = 1.0, 0.8

# --- B1: DETECTED -- 4-run + 2 isolated positives = 6/20 = 30% coverage ---
yB1 = 3.4
statesB1 = [False] * 20
for idx in (5, 6, 7, 8, 12, 15):
    statesB1[idx] = True
x0b1, x1b1 = draw_cells(axB, 0.5, yB1, statesB1, cell_w, cell_h)
start, length = longest_run(statesB1)
run_highlight(axB, 0.5, start, length, yB1, cell_w, cell_h, threshold=3)
pct1 = 100 * sum(statesB1) / len(statesB1)
coverage_bar(axB, x1b1 + 2.0, yB1, cell_h, pct1)
axB.text(x1b1 + 9.5, yB1 + cell_h / 2, 'DETECTED', ha='left', va='center',
          fontsize=11, weight='bold', color=GREEN)

# --- B2: NOT DETECTED -- 7 isolated positives = 35% coverage, but no run >=3 ---
yB2 = 0.6
statesB2 = [False] * 20
for idx in (1, 4, 7, 10, 13, 16, 19):
    statesB2[idx] = True
x0b2, x1b2 = draw_cells(axB, 0.5, yB2, statesB2, cell_w, cell_h)
start2, length2 = longest_run(statesB2)
run_highlight(axB, 0.5, start2, max(length2, 1), yB2, cell_w, cell_h, threshold=3)
pct2 = 100 * sum(statesB2) / len(statesB2)
coverage_bar(axB, x1b2 + 2.0, yB2, cell_h, pct2)
axB.text(x1b2 + 9.5, yB2 + cell_h / 2, 'NOT DETECTED', ha='left', va='center',
          fontsize=11, weight='bold', color=RED)
axB.text(x1b2 + 9.5, yB2 + cell_h / 2 - 0.42, '-- no sustained run', ha='left',
          va='center', fontsize=9.0, color=RED)

axB.set_xlim(*XLIM)
axB.set_ylim(-0.9, 5.3)
axB.axis('off')
axB.add_patch(Rectangle((XLIM[0] + 0.15, -0.8), XLIM[1] - XLIM[0] - 0.3, 6.0,
                          fill=False, edgecolor='black', linewidth=1.4))

# =========================================================================
# PANEL C -- Near-threshold failure (reverse case)
# =========================================================================
axC = fig.add_subplot(gs[2])
axC.set_title('C.  Near-Threshold Failure  (short burst clears neither criterion)',
               fontsize=12.0, weight='bold', loc='left', pad=14)

yC = 1.4
statesC = [False] * 20
statesC[9] = True
statesC[10] = True
x0c, x1c = draw_cells(axC, 0.5, yC, statesC, cell_w, cell_h)

startC, lengthC = longest_run(statesC)
run_highlight(axC, 0.5, startC, lengthC, yC, cell_w, cell_h, threshold=3, label_below=True)

# Dashed reference box showing what a 3-consecutive run would span, positioned
# over the same start so the 1-cell shortfall (cell 11, still white/negative) is visible
ref_x = 0.5 + startC * cell_w
ref_w = 3 * cell_w * 0.92 + (3 - 1) * (cell_w - cell_w * 0.92)
ref = Rectangle((ref_x - 0.14, yC - 0.14), 3 * cell_w - 0.06, cell_h + 0.28,
                 fill=False, edgecolor=GREY, linewidth=1.6, linestyle='--', zorder=7)
axC.add_patch(ref)
axC.text(ref_x + 1.5 * cell_w, yC + cell_h + 0.42, '3-window minimum (reference)',
          ha='center', fontsize=8.0, color=GREY, style='italic')

pctC = 100 * sum(statesC) / len(statesC)
coverage_bar(axC, x1c + 2.0, yC, cell_h, pctC)

axC.text(x1c + 9.5, yC + cell_h / 2, 'NOT DETECTED', ha='left', va='center',
          fontsize=11, weight='bold', color=RED)
axC.text(x1c + 9.5, yC + cell_h / 2 - 0.42, '-- below both thresholds', ha='left',
          va='center', fontsize=9.0, color=RED)

axC.set_xlim(*XLIM)
axC.set_ylim(-0.3, 3.6)
axC.axis('off')
axC.add_patch(Rectangle((XLIM[0] + 0.15, -0.2), XLIM[1] - XLIM[0] - 0.3, 3.6,
                          fill=False, edgecolor='black', linewidth=1.4))

# ---------------------------------------------------------------- caption
fig.suptitle('Event-formation criteria: grouping and detection', fontsize=14.5,
             weight='bold', y=0.995)
fig.text(0.5, 0.005,
         'Event grouping (60 s gap tolerance) and the joint detection criterion '
         '(\u226530% window coverage AND \u22653 consecutive positive windows).',
         ha='center', fontsize=9.8, style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig('event_formation_criteria.png', dpi=200, bbox_inches='tight')
print('saved')
