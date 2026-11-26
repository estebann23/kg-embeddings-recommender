"""Describe cohort ratings affected by the frozen top-100 candidate exclusion.

Run from the repository root: python src/recommenders/rq1_top100_rating_distribution.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/hummus_4000_v1/reviews.csv"
SELECTION = ROOT / "results/rq1_top100_exclusion_test/excluded_top100_training_mean.csv"
OUTPUT = ROOT / "results/rq1_top100_rating_distribution"
STARS = range(1, 6)
SPLITS = ("train", "validation", "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(rows: list[dict[str, str]]) -> dict:
    hist = Counter(int(row["rating"]) for row in rows)
    count = len(rows)
    total = sum(star * hist[star] for star in STARS)
    return {
        "ratings": count,
        "users": len({row["user_id"] for row in rows}),
        "recipes": len({row["recipe_id"] for row in rows}),
        "star_counts": {str(star): hist[star] for star in STARS},
        "star_percent": {str(star): round(100 * hist[star] / count, 4) if count else 0 for star in STARS},
        "mean_stars": round(total / count, 6) if count else None,
        "positive_4_5_percent": round(100 * (hist[4] + hist[5]) / count, 4) if count else 0,
        "rating_sum": total,
    }


def chart_all(summary: dict) -> None:
    counts = [summary["star_counts"][str(s)] for s in STARS]
    fig, ax = plt.subplots(figsize=(9, 5.3))
    bars = ax.bar(STARS, counts, color=["#af4d46", "#d1855c", "#d9bb66", "#7da68c", "#316f70"])
    ax.bar_label(bars, labels=[f"{n:,}\n({100*n/summary['ratings']:.1f}%)" for n in counts], padding=3, fontsize=10)
    ax.set(xlabel="Observed rating (stars)", ylabel="Number of user–recipe ratings",
           title="4,000-user cohort: all 160,000 observed ratings")
    ax.set_xticks(list(STARS))
    ax.set_ylim(0, max(counts) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "all_cohort_ratings.png", dpi=180)
    plt.close(fig)


def chart_top(top: dict, users_by_star: dict[str, int], covered: int, users: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 2.1]})
    axes[0].bar(["Rated ≥1\ntop-100 recipe", "Rated none"], [covered, users - covered],
                color=["#316f70", "#bdc7c9"])
    for index, number in enumerate((covered, users - covered)):
        axes[0].text(index, number + users * .015, f"{number:,}\n({number/users:.1%})", ha="center")
    axes[0].set(ylabel="Distinct cohort users", title="User reach of selected recipes", ylim=(0, users * 1.15))
    counts = [top["star_counts"][str(s)] for s in STARS]
    unique = [users_by_star[str(s)] for s in STARS]
    x = list(STARS)
    width = .39
    a = axes[1].bar([v - width / 2 for v in x], counts, width, label="Ratings", color="#316f70")
    b = axes[1].bar([v + width / 2 for v in x], unique, width, label="Distinct users", color="#d9bb66")
    axes[1].bar_label(a, padding=2, fontsize=8)
    axes[1].bar_label(b, padding=2, fontsize=8)
    axes[1].set(xlabel="Observed rating (stars)", ylabel="Count",
                title="How the selected 100 recipes were rated")
    axes[1].set_xticks(x)
    axes[1].legend(frameon=False)
    fig.text(.5, .01, "Distinct users can appear in multiple star bars. Ratings include train, validation and test.",
             ha="center", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(OUTPUT / "top100_user_reach_and_ratings.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_rows = read_csv(DATA)
    rows = [row for row in source_rows if row["is_rated"] == "True"]
    selected_rows = read_csv(SELECTION)
    selected = {row["recipe_id"] for row in selected_rows}
    assert len(selected_rows) == len(selected) == 100
    assert len(source_rows) == 160001 and len(rows) == 160000
    assert Counter(row["split"] for row in source_rows if row["is_rated"] != "True") == {"unrated": 1}
    assert all(row["is_rated"] == "True" and int(row["rating"]) in STARS for row in rows)
    assert Counter(row["split"] for row in rows) == {"train": 112000, "validation": 16000, "test": 32000}
    users = {row["user_id"] for row in rows}
    assert len(users) == 4000
    assert len({(row["user_id"], row["recipe_id"]) for row in rows}) == len(rows)
    top_rows = [row for row in rows if row["recipe_id"] in selected]
    remainder = [row for row in rows if row["recipe_id"] not in selected]
    top_users = {row["user_id"] for row in top_rows}
    per_user_all = defaultdict(list)
    per_user_kept = defaultdict(list)
    for row in rows:
        per_user_all[row["user_id"]].append(int(row["rating"]))
    for row in remainder:
        per_user_kept[row["user_id"]].append(int(row["rating"]))
    user_changes = {user: statistics.mean(per_user_kept[user]) - statistics.mean(per_user_all[user]) for user in users}
    star_users = {str(star): len({row["user_id"] for row in top_rows if int(row["rating"]) == star}) for star in STARS}

    # Independently reconstruct the selection from training observations.
    training = [row for row in rows if row["split"] == "train"]
    recipe_ratings = defaultdict(list)
    for row in training:
        recipe_ratings[row["recipe_id"]].append(int(row["rating"]))
    ranking = sorted(recipe_ratings, key=lambda rid: (-sum(recipe_ratings[rid]) / len(recipe_ratings[rid]),
                                                     -len(recipe_ratings[rid]), int(rid.rsplit("_", 1)[1])))[:100]
    assert ranking == [row["recipe_id"] for row in selected_rows]
    for row in selected_rows:
        ratings = recipe_ratings[row["recipe_id"]]
        assert sum(ratings) == int(row["training_rating_sum"])
        assert len(ratings) == int(row["training_rating_count"])

    result = {
        "source_sha256": {"reviews": sha256(DATA), "selected_recipes": sha256(SELECTION)},
        "all": summarize(rows), "top100": summarize(top_rows), "without_top100": summarize(remainder),
        "users_rated_top100": len(top_users), "users_rated_top100_percent": round(100 * len(top_users) / len(users), 4),
        "users_by_top100_rating": star_users,
        "top100_ratings_per_reached_user_mean": round(len(top_rows) / len(top_users), 4),
        "top100_share_of_all_ratings_percent": round(100 * len(top_rows) / len(rows), 4),
        "mean_change_without_minus_all": round(summarize(remainder)["mean_stars"] - summarize(rows)["mean_stars"], 6),
        "per_user_mean_effect": {
            "decreased": sum(change < -1e-12 for change in user_changes.values()),
            "increased": sum(change > 1e-12 for change in user_changes.values()),
            "unchanged": sum(abs(change) <= 1e-12 for change in user_changes.values()),
            "mean_change": round(statistics.mean(user_changes.values()), 6),
            "mean_absolute_change": round(statistics.mean(abs(c) for c in user_changes.values()), 6),
            "median_change": round(statistics.median(user_changes.values()), 6),
        },
        "split_comparison": {},
    }
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        removed = [row for row in subset if row["recipe_id"] in selected]
        kept = [row for row in subset if row["recipe_id"] not in selected]
        a, t, k = summarize(subset), summarize(removed), summarize(kept)
        result["split_comparison"][split] = {
            "all": a, "top100": t, "without_top100": k,
            "mean_change_without_minus_all": round(k["mean_stars"] - a["mean_stars"], 6),
            "users_rated_top100": t["users"],
        }
        assert a["ratings"] == t["ratings"] + k["ratings"]
        assert a["rating_sum"] == t["rating_sum"] + k["rating_sum"]
    assert result["all"]["ratings"] == result["top100"]["ratings"] + result["without_top100"]["ratings"]
    assert result["all"]["rating_sum"] == result["top100"]["rating_sum"] + result["without_top100"]["rating_sum"]

    chart_all(result["all"])
    chart_top(result["top100"], star_users, len(top_users), len(users))
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
