
"""
PT1_25config_analysis.py

Analysis and visualization script for the PT_1 25-configuration results.

INPUTS
------
1. all_subject_metrics.csv
2. configuration_summary_25.csv

The two input paths are configurable in the USER SETTINGS section below.

OUTPUTS
-------
- PNG figures for configuration ranking, classifier/modality comparisons,
  class-weighting effects, subject variability, best-configuration counts,
  and accuracy-vs-F1.
- CSV tables containing rankings, paired comparisons, weighting effects,
  modality effects, and subject-level best configurations.
- Excel workbook containing the main analysis tables.
- analysis_report.txt containing an automatically generated textual summary.

IMPORTANT
---------
The script only analyzes the supplied results. It does not retrain models,
change predictions, or introduce any new ML procedure.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# USER SETTINGS — CHANGE THESE PATHS
# ============================================================

INPUT_DIR = Path(r"C:\Users\tiet\Desktop\Paramveer\results")

ALL_SUBJECT_METRICS_FILE = INPUT_DIR / "all_subject_metrics.csv"
CONFIGURATION_SUMMARY_FILE = INPUT_DIR / "configuration_summary_25.csv"

OUTPUT_DIR = INPUT_DIR / "analysis"

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

METRICS = ["accuracy", "f1", "kappa"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1": "F1 score",
    "kappa": "Cohen's Kappa",
}

# Number of top configurations to show in compact subject-level plots.
TOP_N_SUBJECT_PLOTS = 5

# ============================================================
# HELPERS
# ============================================================

def ensure_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def short_config(row):
    """Readable label for plots."""
    feature = row["configuration"].split("__")[0]
    clf = row["classifier"]
    weight = row["weighting"]
    return f"{feature} | {clf} | W-{weight}"


def feature_group(row):
    if row["modality"] == "ET":
        return "ET-10"
    if row["modality"] == "EEG":
        return f"EEG-{int(row['eeg_features'])}"
    return f"EEG-{int(row['eeg_features'])}+ET-10"


def base_feature_config(row):
    """Feature-only configuration, independent of classifier/weighting."""
    return feature_group(row)


def savefig(path, dpi=300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def add_value_labels(ax, values, decimals=3):
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(
                i,
                v,
                f"{v:.{decimals}f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def paired_wilcoxon_or_ttest(x, y):
    """
    Paired comparison.

    Preferred test: Wilcoxon signed-rank when scipy is available.
    Fallback: paired t-test.

    Returns test name, statistic, p-value.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]

    if len(x) < 2:
        return "insufficient data", np.nan, np.nan

    diff = x - y

    # If all differences are zero, the paired comparison is exactly tied.
    if np.allclose(diff, 0):
        return "all paired differences = 0", 0.0, 1.0

    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(
            x, y, zero_method="wilcox", alternative="two-sided"
        )
        # scipy's type stubs can expose generic scalar types here.
        # Explicit conversion through Python float keeps Pylance satisfied.
        stat_value = float(np.asarray(stat).item())
        p_value = float(np.asarray(p).item())
        return "Wilcoxon signed-rank", stat_value, p_value
    except Exception:
        from scipy.stats import ttest_rel
        stat, p = ttest_rel(x, y)
        return "paired t-test", float(stat), float(p)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not ALL_SUBJECT_METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{ALL_SUBJECT_METRICS_FILE}\n\n"
            "Change INPUT_DIR in the USER SETTINGS section."
        )

    if not CONFIGURATION_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{CONFIGURATION_SUMMARY_FILE}\n\n"
            "Change INPUT_DIR in the USER SETTINGS section."
        )

    subject = pd.read_csv(ALL_SUBJECT_METRICS_FILE)
    config = pd.read_csv(CONFIGURATION_SUMMARY_FILE)

    ensure_columns(
        subject,
        [
            "subject", "configuration", "classifier", "weighting",
            "modality", "eeg_features", "et_features",
            "accuracy", "f1", "kappa", "n_samples",
        ],
        "all_subject_metrics.csv",
    )

    ensure_columns(
        config,
        [
            "configuration", "classifier", "weighting", "modality",
            "eeg_features", "et_features", "subjects_evaluated",
            "mean_accuracy", "std_accuracy",
            "mean_f1", "std_f1",
            "mean_kappa", "std_kappa",
        ],
        "configuration_summary_25.csv",
    )

    for col in METRICS:
        subject[col] = pd.to_numeric(subject[col], errors="coerce")

    for col in [
        "mean_accuracy", "std_accuracy",
        "mean_f1", "std_f1",
        "mean_kappa", "std_kappa",
    ]:
        config[col] = pd.to_numeric(config[col], errors="coerce")

    subject["feature_config"] = subject.apply(base_feature_config, axis=1)
    config["feature_config"] = config.apply(base_feature_config, axis=1)

    return subject, config


# ============================================================
# TABLES
# ============================================================

def make_rankings(config):
    ranking_tables = {}

    for metric in METRICS:
        mean_col = f"mean_{metric}"
        std_col = f"std_{metric}"

        r = config[
            [
                "configuration", "classifier", "weighting", "modality",
                "eeg_features", "et_features", "subjects_evaluated",
                "mean_accuracy", "std_accuracy",
                "mean_f1", "std_f1",
                "mean_kappa", "std_kappa",
                "feature_config"
            ]
        ].copy()

        r = r.sort_values(mean_col, ascending=False).reset_index(drop=True)
        r["rank"] = np.arange(1, len(r) + 1)
        r["readable_configuration"] = r.apply(short_config, axis=1)

        # Reorder rank first.
        cols = ["rank"] + [c for c in r.columns if c != "rank"]
        ranking_tables[metric] = r[cols]

    return ranking_tables


def make_weighting_effects(subject):
    """
    Paired subject-level effect of weighting.

    Only XGBoost and SVM have ON/OFF branches in this experiment.
    LDA is deliberately excluded because there is no weighted LDA branch.
    """
    rows = []

    grouped = subject[
        subject["classifier"].isin(["XGBoost", "SVM"])
    ].groupby(["subject", "feature_config", "classifier"])

    for (subj, feature_cfg, clf), g in grouped:
        off = g[g["weighting"] == "OFF"]
        on = g[g["weighting"] == "ON"]

        if len(off) != 1 or len(on) != 1:
            continue

        off = off.iloc[0]
        on = on.iloc[0]

        row = {
            "subject": subj,
            "feature_config": feature_cfg,
            "classifier": clf,
        }

        for metric in METRICS:
            row[f"off_{metric}"] = off[metric]
            row[f"on_{metric}"] = on[metric]
            row[f"delta_{metric}"] = on[metric] - off[metric]

        rows.append(row)

    return pd.DataFrame(rows)


def make_modality_effects(subject):
    """
    Paired subject-level comparison between feature configurations.

    Comparisons:
      EEG-50 -> EEG-50+ET-10
      EEG-75 -> EEG-75+ET-10

    Done separately for each classifier/weighting branch that exists.
    """
    rows = []

    # Match each EEG configuration with its corresponding hybrid configuration.
    feature_pairs = [
        ("EEG-50", "EEG-50+ET-10"),
        ("EEG-75", "EEG-75+ET-10"),
    ]

    for eeg_cfg, hybrid_cfg in feature_pairs:
        a = subject[subject["feature_config"] == eeg_cfg]
        b = subject[subject["feature_config"] == hybrid_cfg]

        keys = ["subject", "classifier", "weighting"]

        merged = a.merge(
            b,
            on=keys,
            suffixes=("_eeg", "_hybrid"),
        )

        for _, r in merged.iterrows():
            row = {
                "subject": r["subject"],
                "classifier": r["classifier"],
                "weighting": r["weighting"],
                "comparison": f"{eeg_cfg} -> {hybrid_cfg}",
            }

            for metric in METRICS:
                row[f"eeg_{metric}"] = r[f"{metric}_eeg"]
                row[f"hybrid_{metric}"] = r[f"{metric}_hybrid"]
                row[f"delta_{metric}"] = (
                    r[f"{metric}_hybrid"] - r[f"{metric}_eeg"]
                )

            rows.append(row)

    return pd.DataFrame(rows)


def make_subject_best(subject):
    """
    For each subject and metric, identify the configuration with the
    highest observed subject-level metric.
    """
    rows = []

    for metric in METRICS:
        idx = subject.groupby("subject")[metric].idxmax()
        best = subject.loc[idx].copy()
        best["metric"] = metric
        best["best_value"] = best[metric]

        rows.append(
            best[
                [
                    "subject", "metric", "configuration",
                    "classifier", "weighting", "modality",
                    "feature_config", "best_value"
                ]
            ]
        )

    return pd.concat(rows, ignore_index=True)


# ============================================================
# FIGURES
# ============================================================

def plot_metric_ranking(config, metric):
    mean_col = f"mean_{metric}"
    std_col = f"std_{metric}"

    d = config.sort_values(mean_col, ascending=True).copy()
    labels = d.apply(short_config, axis=1)

    fig, ax = plt.subplots(figsize=(12, 11))
    y = np.arange(len(d))

    ax.barh(
        y,
        d[mean_col],
        xerr=d[std_col],
        capsize=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_title(
        f"All 25 configurations ranked by {METRIC_LABELS[metric]}"
    )
    ax.grid(axis="x", alpha=0.25)

    for i, v in enumerate(d[mean_col]):
        ax.text(
            v + 0.002,
            i,
            f"{v:.3f}",
            va="center",
            fontsize=7,
        )

    savefig(OUTPUT_DIR / f"01_ranking_{metric}.png")


def plot_classifier_feature_heatmaps(config):
    """
    Heatmaps implemented with matplotlib only.

    One figure per classifier, showing mean F1 for the five feature
    configurations. XGBoost/SVM use Weight-OFF and Weight-ON as columns;
    LDA has only Weight-OFF.
    """
    for clf in ["XGBoost", "SVM", "LDA"]:
        d = config[config["classifier"] == clf].copy()

        pivot = d.pivot_table(
            index="feature_config",
            columns="weighting",
            values="mean_f1",
            aggfunc="mean",
        )

        order = ["ET-10", "EEG-50", "EEG-75", "EEG-50+ET-10", "EEG-75+ET-10"]
        pivot = pivot.reindex([x for x in order if x in pivot.index])

        cols = [x for x in ["OFF", "ON"] if x in pivot.columns]
        pivot = pivot.reindex(columns=cols)

        fig, ax = plt.subplots(figsize=(7, 5))

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            interpolation="nearest",
        )

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("Weighting")
        ax.set_ylabel("Feature configuration")
        ax.set_title(f"{clf}: mean F1 across feature configurations")

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(
                        j, i, f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

        fig.colorbar(im, ax=ax, label="Mean F1")
        savefig(OUTPUT_DIR / f"02_heatmap_f1_{clf.lower()}.png")


def plot_feature_configuration_comparison(config):
    """
    For each feature configuration, compare the three classifiers using
    the best available weighting branch for F1.

    This is explicitly labeled as 'best observed weighting' rather than
    pretending that all classifiers have identical weighting branches.
    """
    rows = []

    for feature_cfg in [
        "ET-10", "EEG-50", "EEG-75",
        "EEG-50+ET-10", "EEG-75+ET-10"
    ]:
        d = config[config["feature_config"] == feature_cfg]

        for clf in ["XGBoost", "SVM", "LDA"]:
            x = d[d["classifier"] == clf]
            if x.empty:
                continue

            best = x.loc[x["mean_f1"].idxmax()]
            rows.append(
                {
                    "feature_config": feature_cfg,
                    "classifier": clf,
                    "mean_f1": best["mean_f1"],
                    "weighting": best["weighting"],
                }
            )

    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 6))
    feature_order = [
        "ET-10", "EEG-50", "EEG-75",
        "EEG-50+ET-10", "EEG-75+ET-10"
    ]
    x = np.arange(len(feature_order))
    width = 0.25

    for i, clf in enumerate(["XGBoost", "SVM", "LDA"]):
        vals = []
        for f in feature_order:
            z = d[(d["feature_config"] == f) & (d["classifier"] == clf)]
            vals.append(z["mean_f1"].iloc[0] if not z.empty else np.nan)

        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=clf,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_order, rotation=20, ha="right")
    ax.set_ylabel("Mean F1")
    ax.set_title("Classifier comparison by feature configuration")
    ax.legend(title="Classifier")
    ax.grid(axis="y", alpha=0.25)

    savefig(OUTPUT_DIR / "03_classifier_feature_comparison_f1.png")

    # Save the underlying table too.
    d.to_csv(
        OUTPUT_DIR / "classifier_feature_comparison_best_weighting.csv",
        index=False,
    )


def plot_weighting_effects(weighting):
    if weighting.empty:
        return

    summary = (
        weighting
        .groupby(["classifier", "feature_config"], as_index=False)
        .agg(
            mean_delta_accuracy=("delta_accuracy", "mean"),
            mean_delta_f1=("delta_f1", "mean"),
            mean_delta_kappa=("delta_kappa", "mean"),
            n=("subject", "count"),
        )
    )

    feature_order = [
        "ET-10", "EEG-50", "EEG-75",
        "EEG-50+ET-10", "EEG-75+ET-10"
    ]

    for metric in METRICS:
        col = f"mean_delta_{metric}"

        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(feature_order))
        width = 0.36

        for i, clf in enumerate(["XGBoost", "SVM"]):
            d = summary[summary["classifier"] == clf].set_index("feature_config")
            vals = [
                d.loc[f, col] if f in d.index else np.nan
                for f in feature_order
            ]

            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                label=clf,
            )

        ax.axhline(0, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_order, rotation=20, ha="right")
        ax.set_ylabel(f"Weighted - Unweighted {METRIC_LABELS[metric]}")
        ax.set_title(
            f"Effect of class weighting on {METRIC_LABELS[metric]}"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        savefig(OUTPUT_DIR / f"04_weighting_effect_{metric}.png")


def plot_subject_variability(subject, config):
    """
    Distribution of subject-level F1 for the top configurations by mean F1.
    """
    top_configs = (
        config.sort_values("mean_f1", ascending=False)
        .head(TOP_N_SUBJECT_PLOTS)["configuration"]
        .tolist()
    )

    d = subject[subject["configuration"].isin(top_configs)].copy()

    ordered = (
        config.set_index("configuration")
        .loc[top_configs]
        .sort_values("mean_f1", ascending=False)
        .index
        .tolist()
    )

    data = [
        d.loc[d["configuration"] == c, "f1"].dropna().values
        for c in ordered
    ]

    labels = []
    for c in ordered:
        r = config.loc[config["configuration"] == c].iloc[0]
        labels.append(short_config(r))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.boxplot(
        data,
        showmeans=True,
    )
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")

    ax.set_ylabel("Subject-level F1")
    ax.set_title(
        f"Subject-level F1 distributions for top {len(ordered)} configurations"
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    savefig(OUTPUT_DIR / "05_subject_f1_distributions_top_configs.png")


def plot_best_configuration_counts(subject_best, metric):
    d = subject_best[subject_best["metric"] == metric].copy()

    counts = (
        d.groupby("configuration")
        .size()
        .sort_values(ascending=True)
    )

    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(counts))

    ax.barh(y, counts.values)
    ax.set_yticks(y)

    labels = []
    for c in counts.index:
        r = d[d["configuration"] == c].iloc[0]
        labels.append(short_config(r))

    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Number of subjects")
    ax.set_title(
        f"Number of subjects for which each configuration was best by {METRIC_LABELS[metric]}"
    )
    ax.grid(axis="x", alpha=0.25)

    savefig(OUTPUT_DIR / f"06_subject_best_counts_{metric}.png")


def plot_accuracy_vs_f1(config):
    d = config.copy()

    fig, ax = plt.subplots(figsize=(9, 7))

    for clf in ["XGBoost", "SVM", "LDA"]:
        z = d[d["classifier"] == clf]
        ax.scatter(
            z["mean_accuracy"],
            z["mean_f1"],
            s=70,
            label=clf,
        )

        for _, r in z.iterrows():
            ax.annotate(
                r["feature_config"],
                (r["mean_accuracy"], r["mean_f1"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                alpha=0.75,
            )

    ax.set_xlabel("Mean accuracy")
    ax.set_ylabel("Mean F1")
    ax.set_title("Configuration-level accuracy vs F1")
    ax.legend(title="Classifier")
    ax.grid(alpha=0.25)

    savefig(OUTPUT_DIR / "07_accuracy_vs_f1.png")


def plot_metric_by_feature_configuration(config):
    """
    Five feature configurations, with mean F1/accuracy/Kappa shown
    as separate figures to avoid mixing metric scales.
    """
    order = [
        "ET-10", "EEG-50", "EEG-75",
        "EEG-50+ET-10", "EEG-75+ET-10"
    ]

    for metric in METRICS:
        mean_col = f"mean_{metric}"

        d = (
            config
            .groupby("feature_config")[mean_col]
            .mean()
            .reindex(order)
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(d))

        ax.bar(x, d.values)
        ax.set_xticks(x)
        ax.set_xticklabels(d.index, rotation=20, ha="right")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(
            f"Average across classifier/weighting branches: {METRIC_LABELS[metric]}"
        )
        ax.grid(axis="y", alpha=0.25)

        add_value_labels(ax, d.values)
        savefig(
            OUTPUT_DIR / f"08_feature_configuration_{metric}.png"
        )


# ============================================================
# STATISTICAL SUMMARIES
# ============================================================

def make_paired_comparisons(subject, config):
    """
    Paired subject-level comparisons between the top F1 configuration
    and every other configuration.

    This is exploratory and is not a multiple-comparison-corrected
    confirmatory analysis.
    """
    top_config = (
        config.loc[config["mean_f1"].idxmax(), "configuration"]
    )

    rows = []

    reference = (
        subject[subject["configuration"] == top_config]
        [["subject"] + METRICS]
        .copy()
    )

    for c in config["configuration"]:
        if c == top_config:
            continue

        other = (
            subject[subject["configuration"] == c]
            [["subject"] + METRICS]
            .copy()
        )

        merged = reference.merge(
            other,
            on="subject",
            suffixes=("_top", "_other"),
        )

        for metric in METRICS:
            x = merged[f"{metric}_top"].values
            y = merged[f"{metric}_other"].values

            test, stat, p = paired_wilcoxon_or_ttest(x, y)

            rows.append(
                {
                    "reference_configuration": top_config,
                    "comparison_configuration": c,
                    "metric": metric,
                    "n_pairs": len(
                        np.asarray(x)[
                            np.isfinite(x) & np.isfinite(y)
                        ]
                    ),
                    "reference_minus_comparison_mean": np.nanmean(x - y),
                    "test": test,
                    "statistic": stat,
                    "p_value": p,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# TEXT REPORT
# ============================================================

def generate_report(subject, config, rankings, weighting, modality, best):
    lines = []

    n_subjects = subject["subject"].nunique()
    n_configs = config["configuration"].nunique()
    n_rows = len(subject)

    lines.append("PT_1 25-CONFIGURATION ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Subjects represented: {n_subjects}")
    lines.append(f"Configurations: {n_configs}")
    lines.append(f"Subject-level result rows: {n_rows}")
    lines.append("")

    if n_subjects != 42:
        lines.append(
            f"NOTE: The supplied result file contains {n_subjects} subjects, "
            "not 42. The configuration summary should therefore be interpreted "
            "as being based on the subjects actually evaluated in this file."
        )
        lines.append("")

    # Overall leaders.
    lines.append("1. TOP CONFIGURATIONS")
    lines.append("-" * 60)

    for metric in METRICS:
        r = rankings[metric].iloc[0]
        lines.append(
            f"{METRIC_LABELS[metric]}: "
            f"{r['readable_configuration']} | "
            f"mean={r[f'mean_{metric}']:.4f} | "
            f"SD={r[f'std_{metric}']:.4f}"
        )

    lines.append("")

    # Top 5 by F1.
    lines.append("2. TOP 5 BY MEAN F1")
    lines.append("-" * 60)

    for _, r in rankings["f1"].head(5).iterrows():
        lines.append(
            f"Rank {int(r['rank'])}: {r['readable_configuration']} | "
            f"F1={r['mean_f1']:.4f} ± {r['std_f1']:.4f} | "
            f"Accuracy={r['mean_accuracy']:.4f} | "
            f"Kappa={r['mean_kappa']:.4f}"
        )

    lines.append("")

    # Best by classifier.
    lines.append("3. BEST CONFIGURATION WITHIN EACH CLASSIFIER")
    lines.append("-" * 60)

    for clf in ["XGBoost", "SVM", "LDA"]:
        d = config[config["classifier"] == clf]
        if d.empty:
            continue
        r = d.loc[d["mean_f1"].idxmax()]
        lines.append(
            f"{clf}: {short_config(r)} | "
            f"F1={r['mean_f1']:.4f} | "
            f"Accuracy={r['mean_accuracy']:.4f} | "
            f"Kappa={r['mean_kappa']:.4f}"
        )

    lines.append("")

    # Weighting.
    lines.append("4. CLASS-WEIGHTING EFFECT")
    lines.append("-" * 60)

    if weighting.empty:
        lines.append("No weighting comparison could be calculated.")
    else:
        for clf in ["XGBoost", "SVM"]:
            d = weighting[weighting["classifier"] == clf]
            if d.empty:
                continue

            lines.append(f"{clf}:")
            for feature_cfg in [
                "ET-10", "EEG-50", "EEG-75",
                "EEG-50+ET-10", "EEG-75+ET-10"
            ]:
                z = d[d["feature_config"] == feature_cfg]
                if z.empty:
                    continue

                lines.append(
                    f"  {feature_cfg}: "
                    f"ΔF1={z['delta_f1'].mean():+.4f}, "
                    f"ΔAccuracy={z['delta_accuracy'].mean():+.4f}, "
                    f"ΔKappa={z['delta_kappa'].mean():+.4f}"
                )

    lines.append("")

    # Hybrid effects.
    lines.append("5. EEG + ET EFFECT")
    lines.append("-" * 60)

    if modality.empty:
        lines.append("No EEG-to-hybrid paired comparisons available.")
    else:
        for comparison in modality["comparison"].unique():
            z = modality[modality["comparison"] == comparison]
            lines.append(f"{comparison}:")
            for clf in ["XGBoost", "SVM", "LDA"]:
                zz = z[z["classifier"] == clf]
                if zz.empty:
                    continue
                # Weighting is naturally included in the grouping here.
                lines.append(
                    f"  {clf}: "
                    f"mean ΔF1={zz['delta_f1'].mean():+.4f}, "
                    f"mean ΔAccuracy={zz['delta_accuracy'].mean():+.4f}, "
                    f"mean ΔKappa={zz['delta_kappa'].mean():+.4f}"
                )

    lines.append("")

    # Subject winners.
    lines.append("6. SUBJECT-LEVEL WINNER COUNTS")
    lines.append("-" * 60)

    for metric in METRICS:
        d = best[best["metric"] == metric]
        counts = d["configuration"].value_counts().head(5)

        lines.append(f"Top 5 winner counts by {METRIC_LABELS[metric]}:")
        for c, count in counts.items():
            r = config[config["configuration"] == c].iloc[0]
            lines.append(
                f"  {short_config(r)}: {int(count)} subjects"
            )
        lines.append("")

    lines.append(
        "Interpretation caution: mean F1/accuracy/Kappa are descriptive "
        "aggregates of subject-level results. Subject heterogeneity is "
        "substantial in this type of analysis, so the ranking should not "
        "be interpreted as proving that one configuration is universally "
        "superior."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subject, config = load_data()

    rankings = make_rankings(config)
    weighting = make_weighting_effects(subject)
    modality = make_modality_effects(subject)
    best = make_subject_best(subject)

    paired = make_paired_comparisons(subject, config)

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    for metric, table in rankings.items():
        table.to_csv(
            OUTPUT_DIR / f"ranking_{metric}.csv",
            index=False,
        )

    weighting.to_csv(
        OUTPUT_DIR / "class_weighting_effects_subject_level.csv",
        index=False,
    )

    modality.to_csv(
        OUTPUT_DIR / "eeg_vs_hybrid_effects_subject_level.csv",
        index=False,
    )

    best.to_csv(
        OUTPUT_DIR / "subject_best_configurations.csv",
        index=False,
    )

    paired.to_csv(
        OUTPUT_DIR / "paired_comparisons_vs_best_f1_config.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    for metric in METRICS:
        plot_metric_ranking(config, metric)

    plot_classifier_feature_heatmaps(config)
    plot_feature_configuration_comparison(config)
    plot_weighting_effects(weighting)
    plot_subject_variability(subject, config)

    for metric in METRICS:
        plot_best_configuration_counts(best, metric)

    plot_accuracy_vs_f1(config)
    plot_metric_by_feature_configuration(config)

    # --------------------------------------------------------
    # Excel workbook
    # --------------------------------------------------------

    excel_path = OUTPUT_DIR / "PT1_25_configuration_analysis.xlsx"

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        config.sort_values("mean_f1", ascending=False).to_excel(
            writer,
            sheet_name="Config_Summary",
            index=False,
        )

        for metric in METRICS:
            rankings[metric].to_excel(
                writer,
                sheet_name=f"Rank_{metric}",
                index=False,
            )

        weighting.to_excel(
            writer,
            sheet_name="Weighting_Effects",
            index=False,
        )

        modality.to_excel(
            writer,
            sheet_name="EEG_vs_Hybrid",
            index=False,
        )

        best.to_excel(
            writer,
            sheet_name="Subject_Winners",
            index=False,
        )

        paired.to_excel(
            writer,
            sheet_name="Paired_Comparisons",
            index=False,
        )

        subject.to_excel(
            writer,
            sheet_name="Subject_Metrics",
            index=False,
        )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    report = generate_report(
        subject,
        config,
        rankings,
        weighting,
        modality,
        best,
    )

    (OUTPUT_DIR / "analysis_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("=" * 70)
    print("PT_1 25-CONFIGURATION ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Subjects represented: {subject['subject'].nunique()}")
    print(f"Configurations:       {config['configuration'].nunique()}")
    print(f"Output directory:     {OUTPUT_DIR}")
    print("")

    for metric in METRICS:
        r = rankings[metric].iloc[0]
        print(
            f"Best {METRIC_LABELS[metric]}: "
            f"{r['readable_configuration']} "
            f"({r[f'mean_{metric}']:.4f})"
        )

    print("")
    print("Main report:")
    print(OUTPUT_DIR / "analysis_report.txt")
    print("Excel workbook:")
    print(excel_path)
    print("")
    print("All figures and CSV tables have been generated.")


if __name__ == "__main__":
    main()
