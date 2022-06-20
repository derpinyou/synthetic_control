#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Набор всех методов и подходов для синтетического контроля результатов
модели."""

import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as sts
import torch
from loguru import logger
from params import Config
from paths import BASE_DIR, DATA_DIR
from utils import read_config, two_step_argparse, log_block


def pivot(deals_dense: pd.DataFrame, target: str) -> pd.DataFrame:
    return deals_dense.reset_index().pivot("dt", "id", target)


def get_theta(x: pd.Series, y: pd.Series) -> float:
    return float(y.cov(x) / x.std()  2)


def cuped(
    config: Config,
    X_pre: pd.DataFrame,
    X_post: pd.DataFrame,
    y_pre: pd.DataFrame,
    y_post: pd.DataFrame,
    all_pre: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Выполнить CUPED над данным.

    :param config: объект :class:Config
    :param X_pre: датафрейм с данными до intervention, не dense
    :param X_post: датафрейм с данными после intervention, не dense 
    :param y_pre: НЕ СХЛОПНУТАЯ в одного группа клиентов
    :param y_post: НЕ СХЛОПНУТАЯ в одного группа клиентов
    :param all_pre: все до даты трита (чтобы точнее подсчитать матожидание)
    :return:
    """
    if not config.control.cuped.enabled:
        logger.info("skipping CUPED as specified in config file")
        return X_pre, X_post, y_pre, y_post
    n_days = config.control.cv.n_future_days
    do_time_based = config.control.cuped.time_based
    if do_time_based:
        X_pre, y_pre = X_pre.copy().iloc[-n_days:], y_pre.copy().iloc[-n_days:]
        X_post, y_post = X_post.copy().iloc[:n_days], y_post.copy().iloc[:n_days]
    axis = 1 if do_time_based else 0
    cov_pre = X_pre.mean(axis=axis)
    theta = 1
    cov_pre_mean = all_pre.mean().mean()
    X_pre_cuped = X_pre.subtract(theta * (cov_pre - cov_pre_mean), axis=1 - axis)
    y_pre_cuped = y_pre.subtract(
        theta * (y_pre.mean(axis=axis) - y_pre.mean(axis=axis).mean(axis=0)),
        axis=1 - axis,
    )
    X_post_cuped = X_post.subtract(
        theta * (cov_pre - cov_pre_mean).values, axis=1 - axis
    )
    y_post_cuped = y_post.subtract(
        theta * (y_pre.mean(axis=axis) - y_pre.mean(axis=axis).mean(axis=0)).values,
        axis=1 - axis,
    )
    for i in (X_pre_cuped, X_post_cuped, y_pre_cuped, y_post_cuped):
        assert not i.isna().any().any()
    return X_pre_cuped, X_post_cuped, y_pre_cuped, y_post_cuped


def split_by_group(
    synth_df: pd.DataFrame, test_group: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбить датафрейм по группе клиентов.

    :param synth_df: кого передать в тестовую группу
    :param test_group: список колонок, которые отправить в тестовую группу
    :return: (контроль, тест)
    """
    test_group = list(test_group)
    return synth_df.drop(columns=test_group), synth_df[test_group]


def split_by_date(
    synth_df: pd.DataFrame, on_date: datetime.date
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разбить датафрейм по дате on_date.

    :param synth_df: датафрейм с данными
    :param on_date: по какой дате бить
    :return:
    """
    return (
        synth_df[synth_df.index.date < on_date],
        synth_df[synth_df.index.date >= on_date],
    )


class MyModel(torch.nn.Module):
    """Хак для линейной регрессии с условием.

    Необходимо :math:\sum_i w_i = 1, :math:w_i > 0, этого нет нигде в sklearn.
    Будем ставить задачу в :mod:pytorch вида

    .. math::
        y = \sum_{i=1}^K \mathrm{sigmoid} (w_i) \cdot x_i

    оптимизируя при этом :math:w_i. Финальным коэффициентом регрессии объявим :math:\mathrm{sigmoid} (\hat{w}_i)
    от найденного оптимума - это сразу даст выполнение обоих условий.
    """

    def init(self, X_pre: np.ndarray, seed: int):
        super().init()
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.w = torch.nn.Parameter(
            torch.normal(
                0,
                0.1,
                (X_pre.shape[1],),
                device="cpu",
                generator=self.generator,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

    def forward(self, x):
        # return torch.matmul(x, self.w)
        return torch.matmul(x, torch.softmax(self.w, dim=-1))

def make_dense_deals_df(
    deals_df: pd.DataFrame,
    min_date: Union[str, datetime.date] = "2021-04-01",
    max_date: str = None,
) -> pd.DataFrame:
    """
    
    :param deals_df: датафрейм с timestamp, account_id, usd_amounts, rub_profits
    :param min_date: минимальная дата, включительно
    :param max_date: максимальная дата, включительно
    :return: датафрейм агрегатов без пробелов по дням
    """
    if max_date is None:
        max_date = deals_df["timestamp"].dt.date.max()
    dates = pd.date_range(min_date, max_date, freq="B")
    users = deals_df["account"].unique()
    idx = pd.MultiIndex.from_product((users, dates), names=["id", "dt"])
    deals_dense = (
        deals_df.groupby(["account", pd.Grouper(key="timestamp", freq="D")])
        .agg({"usd_amount": "sum", "rub_profits": "sum"})
        .reindex(idx, fill_value=0.0)
    )
    return deals_dense

def synthetic_control(
    X_pre: np.ndarray,
    y_pre: np.ndarray,
    lr: float,
    loss_func: str,
    epochs: int = 2000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Обертка для запуска pytorch-оптимизатора :class:MyModel.

    :param X_pre: А-группа до intervention
    :param y_pre: B-группа до intervention
    :param lr: learning rate
    :param loss_func: "mse" или "mae"
    :param epochs: количество эпох обучения
    :param seed: random seed
    :return: пара (numpy массив коэффициентов, loss на оптимуме)
    """

    x = torch.from_numpy(X_pre.astype(np.float32))
    y = torch.from_numpy(y_pre.astype(np.float32)).reshape(y_pre.shape[0])
    if loss_func == "mse":
        crit = torch.nn.MSELoss(reduction="sum")
    elif loss_func == "mae":
        crit = torch.nn.L1Loss(reduction="sum")
    else:
        raise NotImplementedError()
        
    model = MyModel(X_pre, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.randn(())
    for t in range(epochs):
        y_pred = model(x)
        loss = crit(y_pred, y)
        if t % 1000 == 999:
            logger.debug(f"iteration {t} with loss {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (
        torch.softmax(list(model.parameters())[0].T.detach(), dim=-1).numpy(),
        loss.detach().numpy(),
    )


def run(df: pd.DataFrame, config: Config, target: str) -> None:
    """Запускает весь пайплайн по синт.контролю.
    
    :param df: данные
    :param config: объект :class:dp_fx_pricing.params.Config
    :param target: "rub_profits" или "usd_amount"
    """
    with log_block("read deals and prepare dense df"):
        deals_df = df
        logger.warning("dropping bad ids for dataset")
        with log_block("making dense dataframe"):
            dense_deals = make_dense_deals_df(deals_df, min_date=config.ab_start_date)
            synth_df = pivot(deals_dense=dense_deals, target=target)
    with log_block("running group-based cross-validation with sensitivity check"):
        results, (X_pre_d, X_post_d, y_pre_d, y_post_d, reg) = group_based_cv(synth_df)
        rsearch_intercept(results)
        _path = DATA_DIR / "interim" / "group"
        _path.mkdir(exist_ok=True, parents=True)
        results.to_csv(_path / f"cv_results_{target}.csv")
        reg.to_csv(_path / f"regs_{target}.csv")
        _path_data = _path / "splits" / target
        _path_data.mkdir(parents=True, exist_ok=True)
        for seed, data in X_pre_d.items():
            data.to_csv(_path_data / f"x_pre_{seed}.csv")
        for seed, data in X_post_d.items():
            data.to_csv(_path_data / f"x_post_{seed}.csv")
        for seed, data in y_pre_d.items():
            data.to_csv(_path_data / f"y_pre_{seed}.csv")
        for seed, data in y_post_d.items():
            data.to_csv(_path_data / f"y_post_{seed}.csv")
    with log_block("running time-based cross-validation with sensitivity check"):
        results, (X_pre, X_post, y_pre, y_post, reg) = time_based_cv(
            synth_df,
            test_group=None,
        )
        rsearch_intercept(results)
        _path = DATA_DIR / "interim" / "time"
        _path.mkdir(exist_ok=True, parents=True)
        results.to_csv(_path / f"cv_results_{target}.csv")
        X_pre.to_csv(_path / f"x_pre_{target}.csv")
        X_post.to_csv(_path / f"x_post_{target}.csv")
        y_pre.to_csv(_path / f"y_pre_{target}.csv")
        y_post.to_csv(_path / f"y_post_{target}.csv")
        np.save(_path / "reg", reg)


def group_based_cv(
    synth_df: pd.DataFrame, split_date: Optional[datetime.date] = None
) -> Tuple[
    pd.DataFrame,
    Tuple[
        Dict[int, pd.DataFrame],
        Dict[int, pd.DataFrame],
        Dict[int, pd.DataFrame],
        Dict[int, pd.DataFrame],
        pd.DataFrame,
    ],
]:
    """Cross validation на перемешивании групп. Выкидывается группа B, из
    группы A выделяется подмножество мощности B, затем проводится синтетический
    контроль (как в :func:synthetic_control), используя :func:cuped. В
    конце проходит проверка чувствительности с использованием
    :func:sensitivity_check.
    :param synth_df: датафрейм с данными 
    :param split_date: дата сплита (обычно равна ``model_start_date``)
    :return: кортеж (["hue", "mean", "p-value"], где hue равен seed семплинга группы; \
    (словари {seed: X_pre}, {seed: X_post}, {seed: y_pre}, {seed: y_post}, {seed: коэффициенты}) для каждого sample)
    """
    config = read_config()
    if split_date is None:
        split_date = config.control.model_start_date
    rv_x_pre, rv_x_post, rv_y_pre, rv_y_post = {}, {}, {}, {}
    synth_pre, synth_post = split_by_date(synth_df, on_date=split_date)
    X_pre, y_pre = split_by_group(synth_pre, test_group=config.control.treated_clients)
    X_post, y_post = split_by_group(
        synth_post, test_group=config.control.treated_clients
    )
    if config.control.cv.use_cuped:
        _X_pre, _X_post, _y_pre, _y_post = cuped(
            config=config,
            X_pre=X_pre,
            X_post=X_post,
            y_pre=y_pre,
            y_post=y_post,
            all_pre=synth_pre,
        )
    else:
        _X_pre, _X_post, _y_pre, _y_post = (
            X_pre.copy(),
            X_post.copy(),
            y_pre.copy(),
            y_post.copy(),
        )
    _y_pre = _y_pre.mean(axis=1)
    _y_post = _y_post.mean(axis=1)
    rv_x_pre[0] = _X_pre
    rv_x_post[0] = _X_post
    rv_y_pre[0] = _y_pre
    rv_y_post[0] = _y_post
    N_, loss = config.control.n_placebo, "mae"

    results, reg_ = [], None
    regs_array = []
    # Реальная тестовая группа
    with log_block("adding real test group"):
        reg, _ = synthetic_control(
            X_pre=_X_pre.values,
            y_pre=_y_pre.values,
            loss_func="mae",
            lr=config.control.torch.learning_rate,
            epochs=config.control.torch.epochs,
            seed=config.general.random_state,
        )
        results.extend(
            sensitivity_check(
                X_post=_X_post.values, y_post=_y_post, coeffs=reg, label="real"
            )
        )
        regs_array.append((0, list(reg)))
    GROUP_SIZE = len(config.control.treated_clients)
    for i in range(N_):
        seed = config.general.random_state + i
        group = list(X_pre.sample(GROUP_SIZE, axis=1, random_state=seed).columns)
        logger.info(f"placebo with group={group}")
        X_placebo_pre, y_placebo_pre = split_by_group(X_pre, test_group=group)
        X_placebo_pre["mean_treated"] = y_pre.mean(axis=1)
        X_placebo_post, y_placebo_post = split_by_group(X_post, test_group=group)
        X_placebo_post["mean_treated"] = y_post.mean(axis=1)
        if config.control.cv.use_cuped:
            X_placebo_pre, X_placebo_post, y_placebo_pre, y_placebo_post = cuped(
                config=config,
                X_pre=X_placebo_pre,
                X_post=X_placebo_post,
                y_pre=y_placebo_pre,
                y_post=y_placebo_post,
                all_pre=synth_pre,
            )
        y_placebo_pre = y_placebo_pre.mean(axis=1)
        y_placebo_post = y_placebo_post.mean(axis=1)
        rv_x_pre[seed] = X_placebo_pre
        rv_x_post[seed] = X_placebo_post
        rv_y_pre[seed] = y_placebo_pre
        rv_y_post[seed] = y_placebo_post
        reg_, _ = synthetic_control(
            X_placebo_pre.values,
            y_placebo_pre.values,
            lr=config.control.torch.learning_rate,
            epochs=config.control.torch.epochs,
            seed=config.general.random_state,
            loss_func=loss,
        )
        results.extend(
            sensitivity_check(
                X_post=X_placebo_post.values,
                y_post=y_placebo_post.values.reshape(-1),
                coeffs=reg_,
                label=f"fake group with seed={seed}",
                boundary=10 * X_pre.mean().mean(),
            )
        )
        regs_array.append((seed, list(reg_)))
    regs_rv = pd.DataFrame(regs_array, columns=["seed", "coeffs"])
    return pd.DataFrame(results, columns=["hue", "mean", "p_value"]), (
        rv_x_pre,
        rv_x_post,
        rv_y_pre,
        rv_y_post,
        regs_rv,
    )


def sensitivity_check(
    X_post: np.ndarray,
    y_post: np.ndarray,
    coeffs: np.ndarray,
    label: Optional[str] = None,
    boundary: Optional[float] = 1e4,
) -> List[Tuple[str, float, float]]:
    """Проверка чувствительности. В цикле добавляет гауссовый шум и считает
    t-test между :math:coeffs \cdot X и y_post

    :param X_post: A-группа после intervention
    :param y_post: B-группа после intervention
    :param coeffs: коэффициенты регрессии из :meth:synthetic_control
    :param label: для удобства, какую метку поставить в elem[0] каждого из списка
    :param boundary: будет перебирать матожидание от ``-boundary`` до ``boundary``
    :return: список из кортежей вида (label, сколько шума, какой p-value). Нуль соответствует бесшумному случаю.
    """
    config = read_config()
    results = []
    np.random.seed(config.general.random_state)
    for mean in np.linspace(-boundary, boundary, 1000):
        if mean == 0.0:
            continue
        delta = np.random.normal(loc=mean, scale=X_post.mean().mean() * 0.1)
        results.append(
            (label, mean, sts.ttest_ind(np.dot(X_post, coeffs), y_post + delta)[1])
        )
    results.append((label, 0.0, sts.ttest_ind(np.dot(X_post, coeffs), y_post)[1]))
    return results


def time_based_cv(
    synth_df: pd.DataFrame, test_group: Optional[List[str]] = None
) -> Tuple[
    pd.DataFrame,
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray],
]:
    """Кросс-валидация, основанная на перемещении ``model_start_date``.
    Генерируются даты в соответствии с конфигом, затем прогоняется весь
    синт.контроль с измененной ``model_start_date``. Вдобавок проводится оценка
    чувствительности, как в :func:sensitivity_check.

    :param synth_df: датафрейм с данными 
    :param test_group: список id клиентов, входящих в тестовую группу
    :return: сложную пару: (датафрейм ["hue", "mean", "p-value"] где "hue" есть дата сплита; \
    (X_pre, X_post, y_pre, y_post, коэффициенты) - все на истинной дате сплита)
    """
    config = read_config()
    if test_group is None:
        test_group = config.control.treated_clients
    else:

        synth_df = synth_df.drop(columns=config.control.treated_clients)

    results = []
    start, stop, freq = (
        config.control.cv.start_date,
        config.control.cv.stop_date,
        config.control.cv.freq,
    )
    logger.info(
        f"will run through start_date={start}, stop_date={stop} with freq={freq}"
    )
    reg_to_rv = None
    X_pre_rv, X_post_rv, y_pre_rv, y_post_rv = None, None, None, None
    for dt in (
        list(
            pd.date_range(
                start,
                min(
                    stop,
                    config.control.model_start_date
                    - datetime.timedelta(days=config.control.cv.n_future_days),
                ),
                freq=freq,
            )
        )
        + [pd.to_datetime(config.control.model_start_date)]
    ):
        logger.debug(f"handling date {dt}")
        synth_pre, synth_post = split_by_date(synth_df, on_date=dt)

        X_pre, y_pre = split_by_group(synth_pre, test_group=test_group)
        y_pre = y_pre.mean(axis=1)
        X_post, y_post = split_by_group(synth_post, test_group=test_group)
        y_post = y_post.mean(axis=1)
        reg, loss = synthetic_control(
            X_pre.values,
            y_pre.values,
            lr=config.control.torch.learning_rate,
            epochs=config.control.torch.epochs,
            seed=config.general.random_state,
            loss_func="mae",
        )
        if dt == config.control.model_start_date:
            X_pre_rv, X_post_rv, y_pre_rv, y_post_rv, reg_to_rv = (
                X_pre,
                X_post,
                y_pre,
                y_post,
                reg,
            )

        def cond_(x):
            return x[
                x.index.date
                <= dt + datetime.timedelta(days=config.control.cv.n_future_days)
            ]

        np.random.seed(config.general.random_state)
        boundary = 10 * X_pre.mean().mean()
        for mean in np.linspace(-boundary, boundary, 1000):
            if mean == 0.0:
                continue
            delta = np.random.normal(loc=mean, scale=X_post.mean().mean() * 0.1)
            results.append(
                (
                    dt,
                    mean,
                    sts.ttest_ind(np.dot(cond_(X_post), reg), cond_(y_post) + delta)[1],
                )
            )
        results.append(
            (dt, 0.0, sts.ttest_ind(np.dot(cond_(X_post), reg), cond_(y_post))[1])
        )
    return pd.DataFrame(results, columns=["hue", "mean", "p_value"]), (
        X_pre_rv,
        X_post_rv,
        y_pre_rv,
        y_post_rv,
        reg_to_rv,
    )


def rsearch_intercept(df: pd.DataFrame):
    """Ищет правую точку пересечения графика p-value от mean с желаемым alpha в
    конфиге. Это соответствует чувствительности.

    :param df: датафрейм с колонками [x_label, mean, p-value] - \
    дата сплита на cv фолде, матожидание шума и полученный p-value

    :return: тот же df, но на каждую дату добавлена колонка sens
    """
    config = read_config()
    x_label = "hue"
    tmp_ = df.sort_values([x_label, "mean"]).query("mean != -1.0")
    _results = []
    for dt in tmp_[x_label].unique():
        tmp__ = tmp_.loc[tmp_[x_label] == dt]
        for row in tmp__[::-1].itertuples():
            if row.p_value > config.control.alpha:
                break
            mm = row
        _results.append((dt, mm.mean))
    for (dt, sens) in _results:
        df.loc[df[x_label] == dt, "sens"] = sens
    return df


if name == "main":
    r = two_step_argparse(Path(file).name)
    parser = next(r)
    args, _ = next(r)
    conf = read_config()
    for target in ("usd_amounts", "rub_profits"):
        with log_block(f"running target {target}"):
            run(config=conf, target=target)

