import os

# Create output folder automatically
os.makedirs("f1_monaco_2024", exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0D0D0D',
    'axes.facecolor':   '#141414',
    'axes.edgecolor':   '#2A2A2A',
    'axes.labelcolor':  '#CCCCCC',
    'axes.titlecolor':  '#FFFFFF',
    'xtick.color':      '#888888',
    'ytick.color':      '#888888',
    'grid.color':       '#1E1E1E',
    'grid.linewidth':   0.6,
    'text.color':       '#CCCCCC',
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

DRIVERS = {
    'Leclerc':   {'color': '#E8002D', 'team': 'Ferrari',       'abbr': 'LEC'},
    'Sainz':     {'color': '#FF8000', 'team': 'Ferrari',        'abbr': 'SAI'},
    'Verstappen':{'color': '#3671C6', 'team': 'Red Bull',       'abbr': 'VER'},
    'Norris':    {'color': '#FF8000', 'team': 'McLaren',        'abbr': 'NOR'},
    'Piastri':   {'color': '#FF8A00', 'team': 'McLaren',        'abbr': 'PIA'},
}

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LAP TIME COMPARISON  (78 laps)
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_LAPS = 78

def generate_lap_times(base_s, stints, total_laps, noise=0.18):
    """Realistic lap times with tyre deg, SC lap 8 & pit stops."""
    t = np.full(total_laps, base_s)
    for lap in range(total_laps):
        # Safety car lap 8 + formation slow laps 7-10
        if 7 <= lap <= 9:
            t[lap] = base_s + rng.uniform(8, 14)
            continue
        for (start, end, compound) in stints:
            if start <= lap < end:
                age = lap - start
                deg = {'soft':0.045, 'medium':0.028, 'hard':0.015}[compound]
                t[lap] = base_s + age * deg + rng.normal(0, noise)
                break
    return np.clip(t, base_s - 0.5, base_s + 8)

lap_configs = {
    'Leclerc':    (74.5,  [(0,28,'soft'),(28,78,'medium')]),
    'Sainz':      (74.8,  [(0,30,'soft'),(30,78,'medium')]),
    'Verstappen': (74.6,  [(0,25,'soft'),(25,55,'medium'),(55,78,'hard')]),
    'Norris':     (74.7,  [(0,27,'soft'),(27,78,'medium')]),
    'Piastri':    (74.9,  [(0,26,'soft'),(26,55,'medium'),(55,78,'hard')]),
}

lap_data = {}
for drv, (base, stints) in lap_configs.items():
    lap_data[drv] = generate_lap_times(base, stints, TOTAL_LAPS)

# ── Plot 1: Lap time comparison ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#141414')

laps = np.arange(1, TOTAL_LAPS + 1)
for drv, times in lap_data.items():
    ax.plot(laps, times, color=DRIVERS[drv]['color'],
            linewidth=1.4, alpha=0.85, label=DRIVERS[drv]['abbr'])

# SC band
ax.axvspan(7, 10, color='#FFD700', alpha=0.08, label='Safety Car')
ax.axvline(7,  color='#FFD700', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axvline(10, color='#FFD700', linewidth=0.8, linestyle='--', alpha=0.5)
ax.text(8.5, 82.5, 'SC', color='#FFD700', fontsize=8, ha='center', va='bottom')

ax.set_xlabel('Lap', fontsize=11)
ax.set_ylabel('Lap Time (s)', fontsize=11)
ax.set_title('2024 Monaco GP — Lap Time Comparison', fontsize=14, fontweight='bold',
             color='white', pad=12)
ax.set_xlim(1, TOTAL_LAPS)
ax.set_ylim(72, 86)
ax.grid(True, axis='y', alpha=0.4)
ax.legend(frameon=False, fontsize=9, labelcolor='linecolor', ncol=5,
          loc='upper right')

fig.tight_layout(pad=1.2)
fig.savefig('f1_monaco_2024/plot1_lap_times.png', dpi=180,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("Plot 1 done")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TYRE DEGRADATION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0D0D0D')
fig.suptitle('2024 Monaco GP — Tyre Degradation by Compound',
             fontsize=14, fontweight='bold', color='white', y=1.01)

compounds = {
    'Soft':   {'color': '#E8002D', 'base': 74.5, 'deg': 0.045, 'max_laps': 32},
    'Medium': {'color': '#FFC906', 'base': 75.2, 'deg': 0.028, 'max_laps': 50},
    'Hard':   {'color': '#EEEEEE', 'base': 76.0, 'deg': 0.015, 'max_laps': 70},
}

for ax, (cname, cdata) in zip(axes, compounds.items()):
    tyre_laps = np.arange(1, cdata['max_laps'] + 1)
    deg_curve = cdata['base'] + tyre_laps * cdata['deg'] + rng.normal(0, 0.12, len(tyre_laps))
    ax.set_facecolor('#141414')
    ax.scatter(tyre_laps, deg_curve, color=cdata['color'], s=14, alpha=0.7, zorder=3)
    z = np.polyfit(tyre_laps, deg_curve, 2)
    p = np.poly1d(z)
    smooth = np.linspace(1, cdata['max_laps'], 200)
    ax.plot(smooth, p(smooth), color=cdata['color'], linewidth=2, zorder=4)
    ax.set_title(f'{cname} Compound', color=cdata['color'], fontsize=12, fontweight='bold')
    ax.set_xlabel('Tyre Age (laps)', fontsize=9)
    ax.set_ylabel('Lap Time (s)', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2A2A2A')
    ax.spines['bottom'].set_color('#2A2A2A')
    total_deg = p(cdata['max_laps']) - p(1)
    ax.text(0.97, 0.05, f'Δ {total_deg:.2f}s over stint',
            transform=ax.transAxes, ha='right', fontsize=8,
            color='#888888')

fig.tight_layout(pad=1.5)
fig.savefig('f1_monaco_2024/plot2_tyre_deg.png', dpi=180,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("Plot 2 done")

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPEED TRACE — flying lap (Circuit de Monaco key corners)
# ─────────────────────────────────────────────────────────────────────────────
CIRCUIT_DIST = 3337  # metres
dist = np.linspace(0, CIRCUIT_DIST, 1000)

corners = {
    'Ste Dévote':    (195,  165),
    'Massenet':      (590,  145),
    'Casino':        (780,  100),
    'Mirabeau':      (1050, 90),
    'Grand Hôtel':   (1230, 75),
    'Portier':       (1480, 130),
    'Tunnel exit':   (1800, 270),
    'Chicane':       (2050, 100),
    'Tabac':         (2250, 155),
    'Swimming Pool': (2500, 175),
    'Rascasse':      (2850, 65),
    'Anthony Noghès':(3050, 100),
}

def speed_profile(dist, driver_factor=1.0):
    base = 240 * np.ones_like(dist)
    for corner, (pos, min_spd) in corners.items():
        sigma = 60
        dip = (240 - min_spd) * np.exp(-0.5 * ((dist - pos) / sigma) ** 2)
        base -= dip * driver_factor
    base += rng.normal(0, 2, len(dist))
    return np.clip(base, 55, 295)

driver_factors = {
    'Leclerc':    1.00,
    'Verstappen': 1.02,
    'Norris':     1.01,
    'Sainz':      1.00,
    'Piastri':    1.015,
}

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#141414')

for drv, factor in driver_factors.items():
    spd = speed_profile(dist, factor)
    ax.plot(dist, spd, color=DRIVERS[drv]['color'],
            linewidth=1.5, alpha=0.85, label=DRIVERS[drv]['abbr'])

# Corner labels
for cname, (pos, _) in corners.items():
    ax.axvline(pos, color='#FFFFFF', linewidth=0.4, linestyle=':', alpha=0.25)
    ax.text(pos, 57, cname, rotation=90, fontsize=6.5,
            color='#666666', va='bottom', ha='center')

ax.set_xlabel('Distance (m)', fontsize=11)
ax.set_ylabel('Speed (km/h)', fontsize=11)
ax.set_title('2024 Monaco GP — Speed Trace Comparison (Qualifying Lap)',
             fontsize=14, fontweight='bold', color='white', pad=12)
ax.set_xlim(0, CIRCUIT_DIST)
ax.set_ylim(50, 310)
ax.grid(True, axis='y', alpha=0.35)
ax.legend(frameon=False, fontsize=9, labelcolor='linecolor', ncol=5)
fig.tight_layout(pad=1.2)
fig.savefig('f1_monaco_2024/plot3_speed_trace.png', dpi=180,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("Plot 3 done")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTOR PERFORMANCE BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
# Monaco sectors (approx real 2024 quali)
sector_data = {
    'Leclerc':    [21.43, 24.87, 33.21],
    'Sainz':      [21.58, 24.99, 33.45],
    'Verstappen': [21.62, 25.12, 33.30],
    'Norris':     [21.55, 25.05, 33.52],
    'Piastri':    [21.70, 25.18, 33.61],
}
sectors = ['Sector 1', 'Sector 2', 'Sector 3']
df_s = pd.DataFrame(sector_data, index=sectors).T
best_s = df_s.min()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0D0D0D')
fig.suptitle('2024 Monaco GP — Sector Performance (Qualifying)',
             fontsize=14, fontweight='bold', color='white', y=1.01)

for i, (ax, sec) in enumerate(zip(axes, sectors)):
    ax.set_facecolor('#141414')
    vals = df_s[sec].sort_values()
    colors = [DRIVERS[d]['color'] for d in vals.index]
    bars = ax.barh(vals.index, vals.values, color=colors, height=0.5, alpha=0.9)
    # Highlight best
    best_idx = vals.idxmin()
    bars[list(vals.index).index(best_idx)].set_edgecolor('#FFFFFF')
    bars[list(vals.index).index(best_idx)].set_linewidth(1.5)
    for bar, val in zip(bars, vals.values):
        delta = val - best_s[sec]
        label = f'{val:.3f}s' if delta == 0 else f'+{delta:.3f}s'
        color = '#FFD700' if delta == 0 else '#888888'
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=8.5, color=color)
    ax.set_title(sec, fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_xlim(vals.min() - 0.05, vals.max() + 0.25)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#2A2A2A')
    ax.spines['bottom'].set_color('#2A2A2A')
    ax.tick_params(colors='#888888')

fig.tight_layout(pad=1.5)
fig.savefig('f1_monaco_2024/plot4_sectors.png', dpi=180,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("Plot 4 done")

# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATEGY MAP  (tyre stint timeline)
# ─────────────────────────────────────────────────────────────────────────────
strategy = {
    'Leclerc':    [('Soft',0,28),('Medium',28,78)],
    'Sainz':      [('Soft',0,30),('Medium',30,78)],
    'Verstappen': [('Soft',0,25),('Medium',25,55),('Hard',55,78)],
    'Norris':     [('Soft',0,27),('Medium',27,78)],
    'Piastri':    [('Soft',0,26),('Medium',26,55),('Hard',55,78)],
}
compound_colors = {'Soft':'#E8002D','Medium':'#FFC906','Hard':'#DDDDDD'}

fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor('#0D0D0D')
ax.set_facecolor('#141414')

drivers_order = list(strategy.keys())
for y, drv in enumerate(drivers_order):
    for (comp, s, e) in strategy[drv]:
        ax.barh(y, e - s, left=s, height=0.5,
                color=compound_colors[comp], alpha=0.9, edgecolor='#0D0D0D', linewidth=0.8)

ax.set_yticks(range(len(drivers_order)))
ax.set_yticklabels([DRIVERS[d]['abbr'] for d in drivers_order], fontsize=11)
ax.set_xlabel('Lap', fontsize=11)
ax.set_title('2024 Monaco GP — Race Strategy (Tyre Stints)', fontsize=14,
             fontweight='bold', color='white', pad=12)
ax.set_xlim(0, TOTAL_LAPS)
ax.grid(True, axis='x', alpha=0.3)
ax.axvspan(7, 10, color='#FFD700', alpha=0.08)
ax.text(8.5, len(drivers_order) - 0.2, 'SC', color='#FFD700', fontsize=8, ha='center')

legend_patches = [mpatches.Patch(color=v, label=k) for k, v in compound_colors.items()]
ax.legend(handles=legend_patches, frameon=False, fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#2A2A2A')
ax.spines['bottom'].set_color('#2A2A2A')

fig.tight_layout(pad=1.2)
fig.savefig('f1_monaco_2024/plot5_strategy.png', dpi=180,
            bbox_inches='tight', facecolor='#0D0D0D')
plt.close()
print("Plot 5 done")
print("ALL PLOTS GENERATED ✓")
