import datetime
from pathlib import Path
from typing import List

from pydantic import BaseModel


class General(BaseModel):
    data_dir: Path
    random_state: int
    reports_dir: Path
    log_dir: Path


class Ab(BaseModel):
    mid_a_cnum: List[str]
    mid_b_cnum: List[str]
    large_a_cnum: List[str]
    large_b_cnum: List[str]


class Train(BaseModel):
    train_size: float
    short_days_period: int
    long_days_period: int
    first_n_short: int
    first_n_long: int


class Torch(BaseModel):
    epochs: int
    learning_rate: float


class Cv(BaseModel):
    start_date: datetime.date
    stop_date: datetime.date
    freq: str
    n_future_days: int
    use_cuped: bool


class Cuped(BaseModel):
    enabled: bool
    time_based: bool


class Control(BaseModel):
    alpha: float
    torch: "Torch"
    n_placebo: int
    cv: "Cv"
    cuped: "Cuped"
    error_plot_threshold: float
    model_start_date: datetime.date
    treated_clients: List[str]


class Config(BaseModel):
    ab_start_date: datetime.date
    current_date: datetime.date
    general: "General"
    ab: "Ab"
    train: "Train"
    control: "Control"

