#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ast
import datetime
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from params import Config
from paths import BASE_DIR, DATA_DIR
from utils import read_config, two_step_argparse, log_block
from visualization import plot_regression, plot_start_date


def plot_sensitivity(
    interceptions: pd.DataFrame,
    target: str,
    prefix: Path,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    config = read_config()
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(
        data=interceptions, x="mean", y="p_value", hue="hue", style="hue", ax=ax
    )
    ax.grid(True)
    _val = 5_000 if target == "pnl_rub_net" else 25_000
    ax.set_xlim(left=-_val, right=_val)
    ax.set_xlabel("mean")
    ax.set_ylabel("p-value")
    ax.set_title(f"Оценка чувствительности для target={target}")
    ax.hlines(
        config.control.alpha,
        xmin=ax.get_xlim()[0] * 0.95,
        xmax=ax.get_xlim()[1] * 0.95,
        label=f"target alpha = {config.control.alpha}",
        linestyle="--",
    )
    ax.legend()
    return fig, ax


def plot_pvalue_interceptions_hist(
    interceptions: pd.DataFrame,
    target: str,
    prefix: Path,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    interceptions.groupby("hue").apply(lambda x: x["sens"].iloc[0]).hist(
        bins=100, ax=ax
    )
    return fig, ax


def run(config: Config, target: str) -> None:
    """Рисует статистику:

    * чувствительность таргета для group- и time-based CV
    * гистограмму шума, при котором p-value переваливает за alpha

    :param config:
    :param target: pnl_rub_net или size_usd
    :return:
    """
    for _type in ("group", "time"):
        _pref = DATA_DIR / "interim" / _type
        interceptions = pd.read_csv(_pref / f"cv_results_{target}.csv", index_col=0)
        _rep_prefix = BASE_DIR / "reports" / _type
        _rep_prefix.mkdir(parents=True, exist_ok=True)
        with log_block(f"({_type.upper()}) plotting sensitivity statistics"):
            fig, ax = plot_sensitivity(interceptions, target, _rep_prefix)
            fig.savefig(_rep_prefix / f"cv_sensitivity_raw__{target}.pdf")
            fig.clear()
        with log_block(f"({_type.upper()}) plotting sensitivity threshold histogram"):
            fig, ax = plot_pvalue_interceptions_hist(interceptions, target, _rep_prefix)
            fig.savefig(_rep_prefix / f"cv_sensitivity_hist__{target}.pdf")
            fig.clear()
            
        # для time-based надо отдельно читать каждый сплит, т.к. дата тритмента меняется.
        # функция отрисовки хочет коэффициенты, поэтому каждый раз свои читать
        
        if _type == "time":
            X_pre = pd.read_csv(
                _pref / f"x_pre_{target}.csv", index_col=0, parse_dates=[0]
            )
            X_post = pd.read_csv(
                _pref / f"x_post_{target}.csv", index_col=0, parse_dates=[0]
            )
            y_pre = pd.read_csv(
                _pref / f"y_pre_{target}.csv", index_col=0, parse_dates=[0]
            )
            y_post = pd.read_csv(
                _pref / f"y_post_{target}.csv", index_col=0, parse_dates=[0]
            )
            reg = np.load(_pref / "reg.npy")
            with log_block("(TIME) plotting final (real) fold"):
                p_value = interceptions.loc[
                    (interceptions["hue"] == str(config.control.model_start_date))
                    & (interceptions["mean"] == 0.0),
                    "p_value",
                ].iloc[0]
                fig, ax = plot_regression(
                    X_train=X_pre,
                    X_test=X_post,
                    f=lambda x: np.dot(x, reg),
                    y_train=y_pre,
                    y_test=y_post,
                    title=f"p-value = {p_value}",
                    diff_only=True,
                )
                plot_start_date(config.control.model_start_date, ax)
                fig.savefig(_rep_prefix / f"cv_real_fold__{target}.pdf")
    with log_block(f"handling group CV for target={target}"):
        regs = pd.read_csv(
            DATA_DIR / "interim" / "group" / f"regs_{target}.csv", index_col="seed"
        )[["coeffs"]]
        regs["coeffs"] = regs["coeffs"].apply(lambda x: ast.literal_eval(x))
        fig, ax = None, None
        diffs = {}
        
        # функция отрисовки хочет коэффициенты, поэтому каждый раз свои читать
        
        for seed, coeffs in regs.itertuples():
            coeffs = np.array(coeffs)
            _path = DATA_DIR / "interim" / "group" / "splits" / target
            X_pre = pd.read_csv(
                _path / f"x_pre_{seed}.csv", index_col=0, parse_dates=[0]
            )
            X_post = pd.read_csv(
                _path / f"x_post_{seed}.csv", index_col=0, parse_dates=[0]
            )
            y_pre = pd.read_csv(
                _path / f"y_pre_{seed}.csv", index_col=0, parse_dates=[0]
            )
            y_post = pd.read_csv(
                _path / f"y_post_{seed}.csv", index_col=0, parse_dates=[0]
            )
            with log_block(f"plotting raw target vs synth target"):
                fig, ax = plot_regression(
                    X_train=X_pre,
                    X_test=X_post,
                    f=lambda x: np.dot(x, coeffs),
                    y_train=y_pre,
                    y_test=y_post,
                    id_=seed,
                    title=f"Регрессии на group CV на target={target}",
                    diff_only=True,
                    fig=fig,
                    ax=ax,
                )
                
            # когда оценивали чувствительность, брали n_future_days
            # будет консистентно брать столько же и при оценке p-value
            
            target_date = config.control.model_start_date + datetime.timedelta(
                days=config.control.cv.n_future_days
            )
            with log_block(f"p-value by plot for seed={seed} on date={target_date}"):
                diffs[seed] = (
                    y_post.loc[str(target_date)]
                    - np.dot(X_post.loc[str(target_date)], coeffs)
                ).iloc[0]
        _tmp = pd.DataFrame(
            [
                value < diffs[0] if diffs[0] < 0 else value > diffs[0]
                for value in diffs.values()
            ],
            index=diffs.keys(),
        )
        p_value_plot = (_tmp.sum() / _tmp.shape[0]).iloc[0]
        logger.info(f"p-value is: {p_value_plot}")

        plot_start_date(config.control.model_start_date, ax=ax)


if name == "main":
    r = two_step_argparse(Path(file).name)
    parser = next(r)
    args, _ = next(r)
    conf = read_config()
    for target in ("rub_profits"):
        with log_block(f"running target {target}"):
            run(config=conf, target=target)

