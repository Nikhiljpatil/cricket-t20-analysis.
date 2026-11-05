"""
Assignment 12 - Mini Project: Data Analysis & Visualization
Dataset : T20 International batting career statistics (twb.csv)
Tools   : Pandas (cleaning), NumPy (numerical analysis), Matplotlib/Seaborn (graphs)

Run from the project folder:   python assignment12_t20_analysis.py
All paths are relative to this script's location.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SOURCE_CSV = BASE / "data" / "twb.csv"
CLEAN_CSV = BASE / "cleaned_cricket_dataset.csv"
SUMMARY_CSV = BASE / "numerical_analysis_summary.csv"
CORR_CSV = BASE / "correlation_matrix.csv"
QUALITY_CSV = BASE / "data_quality_checks.csv"
GRAPH_DIR = BASE / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False})

# ---------------------------------------------------------------- 1. LOAD
print("=" * 70, "\n1. LOADING DATA\n" + "=" * 70)
try:
    raw = pd.read_csv(SOURCE_CSV)
except FileNotFoundError:
    sys.exit(f"ERROR: source file not found: {SOURCE_CSV}")
print(f"Loaded {SOURCE_CSV.name}: {raw.shape[0]} rows x {raw.shape[1]} columns")

# ---------------------------------------------------------------- 2. INSPECT
print("\n" + "=" * 70, "\n2. INSPECTION (before cleaning)\n" + "=" * 70)
print("Columns:", list(raw.columns))
print("\nData types:\n", raw.dtypes.to_string())
print("\nSample records:\n", raw.head(5).to_string())


def quality_report(frame, label):
    """Count simple data-quality problems so before/after can be compared."""
    text_cols = frame.select_dtypes(include=["object", "string"]).columns
    ws = sum(int((frame[c].dropna() != frame[c].dropna().str.strip()).sum()) for c in text_cols)
    blanks = sum(int((frame[c].dropna().str.strip() == "").sum()) for c in text_cols)
    return {
        "stage": label, "rows": len(frame), "columns": frame.shape[1],
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_players": int(frame["Player"].duplicated().sum()),
        "unnamed_index_columns": int(sum(str(c).startswith("Unnamed") for c in frame.columns)),
        "text_cells_with_extra_whitespace": ws, "blank_text_cells": blanks,
    }


before = quality_report(raw, "before_cleaning")

# ---------------------------------------------------------------- 3. CLEAN
print("\n" + "=" * 70, "\n3. CLEANING\n" + "=" * 70)
df = raw.copy()
log = []

# 3a. Drop exported index / fully-empty columns (justified: carry no information)
drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or df[c].isna().all()]
df = df.drop(columns=drop_cols)
log.append(f"Dropped index/empty columns: {drop_cols if drop_cols else 'none found'}")

# 3b. Trim whitespace in column names and text; blank strings -> NaN
df.columns = [str(c).strip() for c in df.columns]
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].str.strip().replace("", np.nan)
log.append("Trimmed whitespace in column names and text cells; blank strings -> NaN")

# 3c. Exact duplicate rows
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
log.append(f"Removed {n_dup} exact duplicate rows")

# 3d. Missing values: only drop rows missing the key fields; nothing is filled in
n_missing = int(df.isna().sum().sum())
key_fields = ["Player", "Mat", "Inns", "Runs", "BF", "SR"]
rows_missing_key = int(df[key_fields].isna().any(axis=1).sum())
df = df.dropna(subset=key_fields).reset_index(drop=True)
log.append(f"Missing values found: {n_missing}; rows dropped for missing key fields: {rows_missing_key}; no values were imputed")

# 3e. Consistency checks (report; correct only if provably wrong)
checks = {
    "NO > Inns": int((df["NO"] > df["Inns"]).sum()),
    "Inns > Mat": int((df["Inns"] > df["Mat"]).sum()),
    "negative counts": int((df[["Mat", "Inns", "NO", "Runs", "BF", "100", "50", "0", "4s", "6s"]] < 0).sum().sum()),
    "Ave != Runs/(Inns-NO) (tol 0.02)": int(((df["Runs"] / (df["Inns"] - df["NO"])).round(2) - df["Ave"]).abs().gt(0.02).sum()),
    "SR != Runs/BF*100 (tol 0.02)": int(((df["Runs"] / df["BF"] * 100).round(2) - df["SR"]).abs().gt(0.02).sum()),
    "boundary runs > total runs": int(((4 * df["4s"] + 6 * df["6s"]) > df["Runs"]).sum()),
}
for k, v in checks.items():
    print(f"  consistency check - {k}: {v} violations")
log.append("Consistency checks (NO<=Inns<=Mat, Ave, SR, boundary runs): " +
           ("no invalid values found, so no values were corrected" if sum(checks.values()) == 0
            else f"violations found {checks}"))

# 3f. Rename numeric-looking columns to descriptive names (original names kept in README)
rename_map = {"100": "Hundreds", "50": "Fifties", "0": "Ducks", "4s": "Fours", "6s": "Sixes"}
df = df.rename(columns=rename_map)
log.append(f"Renamed columns for clarity: {rename_map}")

# 3g. Helper columns (original columns are kept untouched)
df["Name"] = df["Player"].str.replace(r"\s*\(.*\)$", "", regex=True)
df["Team_Raw"] = df["Player"].str.extract(r"\((.*)\)$")[0]
df["HS_NotOut"] = df["HS"].str.endswith("*")                        # asterisk = not out
df["HS_Score"] = df["HS"].str.replace("*", "", regex=False).astype(int)
df["Debut_Year"] = df["Span"].str.split("-").str[0].astype(int)
df["Last_Year"] = df["Span"].str.split("-").str[1].astype(int)
df["Career_Years"] = df["Last_Year"] - df["Debut_Year"]
df["Boundary_Runs"] = 4 * df["Fours"] + 6 * df["Sixes"]
df["Boundary_Pct"] = (df["Boundary_Runs"] / df["Runs"] * 100).round(2)
df["Sixes_per_Inns"] = (df["Sixes"] / df["Inns"]).round(3)
log.append("Added helper columns: Name, Team_Raw, HS_NotOut, HS_Score (asterisk kept in original HS), "
           "Debut_Year, Last_Year, Career_Years, Boundary_Runs, Boundary_Pct, Sixes_per_Inns")

after = quality_report(df, "after_cleaning")
quality = pd.DataFrame([before, after])
quality.to_csv(QUALITY_CSV, index=False)
df.to_csv(CLEAN_CSV, index=False)
print("\nCleaning log:")
for item in log:
    print("  -", item)
print("\nBefore/after quality checks:\n", quality.T.to_string(header=False))
print(f"\nSaved cleaned data -> {CLEAN_CSV.name} ({df.shape[0]} rows x {df.shape[1]} columns)")

# ---------------------------------------------------------------- 4. NUMPY ANALYSIS
print("\n" + "=" * 70, "\n4. NUMERICAL ANALYSIS (NumPy)\n" + "=" * 70)
stat_cols = ["Mat", "Inns", "Runs", "BF", "Ave", "SR", "Hundreds", "Fifties", "Fours", "Sixes", "Boundary_Pct"]
rows = []
for col in stat_cols:
    v = df[col].to_numpy(dtype=float)
    rows.append({"attribute": col, "count": v.size, "mean": np.mean(v), "median": np.median(v),
                 "minimum": np.min(v), "maximum": np.max(v), "total": np.sum(v),
                 "std_dev_sample": np.std(v, ddof=1),
                 "player_at_minimum": df.loc[int(np.argmin(v)), "Name"],
                 "player_at_maximum": df.loc[int(np.argmax(v)), "Name"]})
summary = pd.DataFrame(rows).round(3)
summary.to_csv(SUMMARY_CSV, index=False)
print(summary.drop(columns=["player_at_minimum", "player_at_maximum"]).to_string(index=False))
print(f"\nSaved -> {SUMMARY_CSV.name}")

corr_cols = ["Mat", "Runs", "Ave", "SR", "Hundreds", "Fifties", "Fours", "Sixes", "Boundary_Pct"]
corr = pd.DataFrame(np.corrcoef(df[corr_cols].to_numpy(dtype=float), rowvar=False),
                    index=corr_cols, columns=corr_cols).round(3)
corr.to_csv(CORR_CSV)
print("\nCorrelation matrix (np.corrcoef, Pearson):\n", corr.to_string())
r_runs_sr = corr.loc["Runs", "SR"]
r_runs_mat = corr.loc["Runs", "Mat"]
r_sixes_sr = corr.loc["Sixes", "SR"]
r_runs_ave = corr.loc["Runs", "Ave"]
print(f"\nSaved -> {CORR_CSV.name}")

# ---------------------------------------------------------------- 5. GRAPHS
print("\n" + "=" * 70, "\n5. GRAPHS\n" + "=" * 70)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", GRAPH_DIR.name + "/" + name)


# Graph 1 - top 10 by runs
top_runs = df.nlargest(10, "Runs").iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(top_runs["Name"], top_runs["Runs"], color=BLUE)
ax.bar_label(bars, padding=3, fontsize=10)
ax.set_title("Top 10 T20I Batsmen by Career Runs")
ax.set_xlabel("Career runs"); ax.set_ylabel("Player")
ax.set_xlim(0, top_runs["Runs"].max() * 1.12)
finish(fig, "graph1_top10_runs.png")

# Graph 2 - top 10 by strike rate
top_sr = df.nlargest(10, "SR").iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(top_sr["Name"], top_sr["SR"], color=ORANGE)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=10)
ax.axvline(df["SR"].mean(), color=GRAY, linestyle="--", linewidth=1.2)
ax.text(df["SR"].mean() + 0.8, 9.55, f"dataset mean = {df['SR'].mean():.1f}", color=GRAY, fontsize=9)
ax.set_title("Top 10 T20I Batsmen by Career Strike Rate")
ax.set_xlabel("Strike rate (runs per 100 balls)"); ax.set_ylabel("Player")
ax.set_xlim(0, top_sr["SR"].max() * 1.12)
finish(fig, "graph2_top10_strike_rate.png")

# Graph 3 - histogram of career runs
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(df["Runs"], bins=12, color=BLUE, edgecolor="white")
ax.axvline(df["Runs"].mean(), color=ORANGE, linewidth=2, label=f"Mean = {df['Runs'].mean():.0f}")
ax.axvline(df["Runs"].median(), color=GRAY, linewidth=2, linestyle="--", label=f"Median = {df['Runs'].median():.0f}")
ax.set_title("Distribution of Career Runs (105 T20I Batsmen)")
ax.set_xlabel("Career runs"); ax.set_ylabel("Number of players")
ax.legend(frameon=False)
finish(fig, "graph3_runs_histogram.png")

# Graph 4 - scatter runs vs strike rate with least-squares line
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.scatter(df["Runs"], df["SR"], color=BLUE, alpha=0.75, s=45, edgecolor="white", linewidth=0.6)
slope, intercept = np.polyfit(df["Runs"], df["SR"], 1)
xs = np.linspace(df["Runs"].min(), df["Runs"].max(), 100)
ax.plot(xs, slope * xs + intercept, color=ORANGE, linewidth=2, label=f"Trend line (r = {r_runs_sr:.2f})")
label_offsets = {0: (-30, -30), 1: (-45, -2), 2: (-30, 24)}   # separate the three close labels
for rank, (_, r) in enumerate(df.nlargest(3, "Runs").iterrows()):
    ax.annotate(r["Name"], (r["Runs"], r["SR"]), textcoords="offset points", xytext=label_offsets[rank],
                ha="right", fontsize=9, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
ax.set_title("Career Runs vs Strike Rate")
ax.set_xlabel("Career runs"); ax.set_ylabel("Strike rate (runs per 100 balls)")
ax.legend(frameon=False, loc="lower right")
finish(fig, "graph4_runs_vs_strike_rate.png")

# Graph 5 - correlation heatmap (diverging, centred on 0)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.diverging_palette(250, 30, as_cmap=True, center="light"),
            vmin=-1, vmax=1, center=0, square=True, linewidths=1, linecolor="white",
            cbar_kws={"label": "Pearson correlation (r)", "shrink": 0.8}, annot_kws={"size": 9}, ax=ax)
ax.set_title("Correlation Matrix of Batting Statistics")
ax.tick_params(axis="x", rotation=45)
finish(fig, "graph5_correlation_heatmap.png")

# ---------------------------------------------------------------- 6. FINDINGS
print("\n" + "=" * 70, "\n6. KEY FINDINGS (from this run)\n" + "=" * 70)
best_runs = df.loc[df["Runs"].idxmax()]
best_sr = df.loc[df["SR"].idxmax()]
best_ave = df.loc[df["Ave"].idxmax()]
most_sixes = df.loc[df["Sixes"].idxmax()]
top10_runs_names = set(df.nlargest(10, "Runs")["Name"])
top10_sr_names = set(df.nlargest(10, "SR")["Name"])
print(f"- Players analysed: {len(df)}; total runs {int(df['Runs'].sum()):,}; mean career runs {df['Runs'].mean():.1f} (median {df['Runs'].median():.0f})")
print(f"- Most runs: {best_runs['Name']} ({best_runs['Runs']}); highest strike rate: {best_sr['Name']} ({best_sr['SR']}); "
      f"best average: {best_ave['Name']} ({best_ave['Ave']}); most sixes: {most_sixes['Name']} ({most_sixes['Sixes']})")
print(f"- Players in both top-10 lists: {sorted(top10_runs_names & top10_sr_names) or 'none'}")
print(f"- r(Runs, SR) = {r_runs_sr:.2f}; r(Runs, Mat) = {r_runs_mat:.2f}; r(Runs, Ave) = {r_runs_ave:.2f}; r(Sixes, SR) = {r_sixes_sr:.2f}")
print(f"- Players with a not-out highest score: {int(df['HS_NotOut'].sum())}; centuries in dataset: {int(df['Hundreds'].sum())}")
print("\nOUTPUT FILES")
for p in [CLEAN_CSV, SUMMARY_CSV, CORR_CSV, QUALITY_CSV]:
    print("  ", p.relative_to(BASE))
for p in sorted(GRAPH_DIR.glob("*.png")):
    print("  ", p.relative_to(BASE))
print("\nDone.")
