# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:27:11 2022

@author: Kaley D Boggs
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

COLUMNS = [
    # "Variable",
    "Status (In/Out)",
    "Variable Name",
    "Geo/national",
    "Brand",
    "SubBrand",
    "Variable Type (Media/Base/Trade)",
    "Driver Type",
    "Metric / Units",
    "Support Multiplier",
    "Classification Split",
    "Primary/Secondary",
    "Type of effect (Own/Halo/Comp)",
    "Media FSI adjustment factor Y1",
    "Media FSI adjustment factor Y2",
    "Rho default",
    "Rho lower bound",
    "Rho upper bound",
    "Rho step size",
    "Beta default",
    "Beta lower bound",
    "Beta upper bound",
    "Beta step size",
    "S default",
    "S lower bound",
    "S upper bound",
    "S step size",
    "ROI prior lower bound",
    "ROI prior upper bound",
    "Data Type (Continuous/Categorical)",
    "NOTES",
]


REQUIRED_COLUMNS = [
    # "Variable",
    "Status (In/Out)",
    # "Variable Name",
    "Geo/national",
    "Brand",
    "SubBrand",
    "Variable Type (Media/Base/Trade)",
    # "Metric / Units",
    # "Support Multiplier",
    "Primary/Secondary",
    # "Type of effect (Own/Halo/Comp)",
    "Data Type (Continuous/Categorical)",
]


def support_multiplier_from_metric(metric: str) -> int:
    """
    Simple method to get support multipliers from metric.
    If it is unknown, return 1.
    """
    support_type_to_multiplier = {
        "grps": 1,
        "impressions": 1000000,
        "circulations": 1000000,
        "activations": 1000,
        "prints": 1000000,
        "clicks": 1000,
    }
    return support_type_to_multiplier.get(metric.lower(), 1)


def make_config_group_template(
    Variables: List[str],
    group_values: Dict[str, Any] = {},
) -> pd.DataFrame:
    """
    Simple function to set config values for a given block of variables.
    """
    df = pd.DataFrame(columns=COLUMNS).set_index("Variable Name")

    # Create rows for block
    for v in Variables:
        df.loc[v, :] = np.nan

    # Fill some defaults where possible
    defaults = {
        "Status (In/Out)": "In",
        "Geo/national": "National",
        "Primary/Secondary": "Primary",
        "Type of effect (Own/Halo/Comp)": "Own",
        "Data Type (Continuous/Categorical)": "Continuous",
    }

    # Add default values to group_values, preferring those explicitly defined in
    # group_values
    group_values = {**defaults, **group_values}

    # Infer support multiplier if omitted and metric known
    if (
        "Metric / Units" in group_values.keys()
        and "Support Multiplier" not in group_values.keys()
    ):
        group_values["Support Multiplier"] = support_multiplier_from_metric(
            group_values["Metric / Units"]
        )

#not working
    # Set block-wide cfg values
    #for k, v in group_values.items():
     #   if k not in COLUMNS:
      #      raise ValueError(f"Field: '{k}' not a valid config field")

      #  df.loc[:, k] = v

    #return df


def save_config(df: pd.DataFrame, path: Path) -> None:
    """
    Simple way to write config df to excel with correct column order
    """
    if not df.index.is_unique:
        raise ValueError("Some variables have duplicate entries!")

    df.reset_index()[COLUMNS].to_excel(path, index=False)


def read_config(path: Path, model: str = "Primary") -> Dict[str, Dict[str, Any]]:
    """
    Read and parse xlsx config file to dict
    """

    def _isnan(obj) -> bool:
        """
        Ultimate hack
        """
        if isinstance(obj, (int, float)):
            return np.isnan(obj)
        return False

#need to have the excel to put here
    cfg = pd.read_excel(path).set_index("Variable Name").to_dict(orient="index")

    # Filter out all keys in column dicts if value is np.nan
    cfg = {col: {k: v for k, v in d.items() if not _isnan(v)} for col, d in cfg.items()}

    # Ensure required keys are defined
    for col, d in cfg.items():
        missing_keys = [c for c in REQUIRED_COLUMNS if c not in d.keys()]
        if missing_keys:
            raise LookupError(f"Column {col} missing required keys: {missing_keys}")

    # Filter to "Status In/Out" == "In", model type
    cfg = {
        col: d
        for col, d in cfg.items()
        if d["Status (In/Out)"] == "In" and d["Primary/Secondary"] == model
    }

    return cfg


def get_primary_model_ranges_from_config(
    cfg: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Parse config for primary model estimation
    """
    media_cfg = {
        k: v
        for k, v in cfg.items()
        if v["Variable Type (Media/Base/Trade)"] == "Media"
        and v["Primary/Secondary"] == "Primary"
    }

    media_cfg = {
        c: {
            "rho": {
                "default": d["Rho default"],
                "lower_bound": d["Rho lower bound"],
                "upper_bound": d["Rho upper bound"],
                "step_size": d["Rho step size"],
            },
            "beta": {
                "default": d["Beta default"],
                "lower_bound": d["Beta lower bound"],
                "upper_bound": d["Beta upper bound"],
                "step_size": d["Beta step size"],
            },
            "s": {
                "default": d["S default"],
                "lower_bound": d["S lower bound"],
                "upper_bound": d["S upper bound"],
                "step_size": d["S step size"],
            },
        }
        for c, d in media_cfg.items()
    }

    return media_cfg


def get_categorical_columns_from_config(cfg: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Get list of columns which are categorical / are binary values.
    """
    return [
        c
        for c, d in cfg.items()
        if d["Data Type (Continuous/Categorical)"] == "Categorical"
