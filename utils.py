#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yaml
from src.params import Config
from pathlib import Path
from src.paths import MODULE_DIR
from contextlib import contextmanager
from loguru import logger
from functools import wraps
import time


def read_config(path: Path = MODULE_DIR.parent / "params.yaml") -> Config:
    with open(path, "r") as f:
        conf = yaml.safe_load(f)
    return Config(**conf)

def two_step_argparse(log_name):
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    yield parser
    args = parser.parse_args()
    with open(args.config, "r") as f:
        conf = yaml.safe_load(f)
    log_dir = conf["general"]["log_dir"]

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.opt(exception=True, colors=True)
    if log_dir:
        logger.add(Path(log_dir) / (str(log_name) + "_{time}.log"))
    yield args, conf
    
@contextmanager
def log_block(message: str, depth=2):
    logger_ = logger.opt(depth=depth)
    logger_.info(message)
    yield
    logger_.info(f"(DONE) {message}")


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        print(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

