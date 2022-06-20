#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline


def plot_start_date(model_start_date: datetime.date, ax: plt.Axes) -> None:
    ax.vlines(
        x=model_start_date,
        ymin=ax.get_ylim()[0],
        ymax=ax.get_ylim()[1],
        label="start model",
        linestyles=":",
        lw=2,
    )


def plot_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    f: Optional[Callable[[pd.DataFrame], Union[pd.DataFrame, np.ndarray]]],
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    diff_only: bool = False,
    id_: str = "",
    title: str = "",
    plot_alpha: float = 0.4,
    fig=None,
    ax=None,
) -> Tuple[plt.Figure, plt.Axes]:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    all_dt = pd.concat((X_train, X_test)).index


    y_test = y_test.values.reshape(-1)
    y_train = y_train.values.reshape(-1)

    if diff_only:
        y_real = np.concatenate((y_train, y_test))
        y_pred = np.concatenate((f(X_train), f(X_test)))
        ax.plot(
            all_dt, y_real - y_pred, label=f"real - pred | id {id_}", alpha=plot_alpha
        )
    else:
        y = np.concatenate((y_train, y_test))
        ax.plot(all_dt, y, label=f"real | id {id_}", alpha=plot_alpha)
        y = np.concatenate((f(X_train), f(X_test)))
        ax.plot(all_dt, y, label=f"synth | id {id_}", alpha=plot_alpha)
    ax.legend()
    ax.set_title(title)
    ax.grid(True)
    return fig, ax

