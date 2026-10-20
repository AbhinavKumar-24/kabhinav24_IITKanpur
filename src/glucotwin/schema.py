"""Canonical tables shared by both tracks (column details in docs/data_dictionary.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PROFILE_FEATURES = ["hba1c_pct", "fasting_glucose", "homa_ir", "bmi", "age"]
GROUPS = ("healthy", "prediabetes", "t2d")

CGM_COLUMNS = ["pid", "time", "glucose", "sensor", "censored"]
ACTIVITY_COLUMNS = ["pid", "time", "hr", "mets", "dur_min"]
MEAL_COLUMNS = [
    "pid",
    "time",
    "meal_type",
    "carbs",
    "protein",
    "fat",
    "fiber",
    "calories",
    "fraction",
    "carbs_eaten",
    "protein_eaten",
    "fat_eaten",
    "fiber_eaten",
]
SLEEP_COLUMNS = ["pid", "night", "sleep_hours", "coverage", "available_time", "src", "rmssd"]


class ValidationError(RuntimeError):
    """A stop rule failed (APPROACH.md §2.1). The pipeline must not continue."""


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail" | "info"
    detail: str


@dataclass
class TrackData:
    """Everything one track needs. The two tracks are never mixed in one TrackData."""

    track: str  # "evidence" or "synthetic"
    profiles: pd.DataFrame  # one row per participant, indexed by pid
    cgm: pd.DataFrame  # native sensor readings (both sensors when available)
    activity: pd.DataFrame
    meals: pd.DataFrame
    sleep: pd.DataFrame
    checks: list[Check] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add_check(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def included(self) -> list[str]:
        prof = self.profiles
        if "included" not in prof.columns:
            return list(prof.index)
        return list(prof.index[prof["included"].astype(bool)])

    def subset(self, pids: list[str]) -> TrackData:
        keep = set(pids)
        return TrackData(
            track=self.track,
            profiles=self.profiles.loc[[p for p in self.profiles.index if p in keep]].copy(),
            cgm=self.cgm[self.cgm["pid"].isin(keep)].copy(),
            activity=self.activity[self.activity["pid"].isin(keep)].copy(),
            meals=self.meals[self.meals["pid"].isin(keep)].copy(),
            sleep=self.sleep[self.sleep["pid"].isin(keep)].copy(),
            checks=list(self.checks),
            meta=dict(self.meta),
        )

    def truncated(self, pid: str, until: pd.Timestamp) -> TrackData:
        """One participant's data with everything after ``until`` deleted (replay tests)."""
        sub = self.subset([pid])
        sub.cgm = sub.cgm[sub.cgm["time"] <= until]
        sub.activity = sub.activity[sub.activity["time"] <= until]
        sub.meals = sub.meals[sub.meals["time"] <= until]
        sub.sleep = sub.sleep[sub.sleep["available_time"] <= until]
        return sub


def provenance(track: str, field_name: str) -> str:
    """Display provenance of a profile field: observed, derived or synthetic."""
    if track == "synthetic":
        return "synthetic"
    if field_name in {"homa_ir", "tg_hdl", "group"}:
        return "derived"
    return "observed"
