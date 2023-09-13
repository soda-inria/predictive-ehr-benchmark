from datetime import datetime
import io
import os
from pathlib import Path
import subprocess
from typing import List, Union
from loguru import logger

import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import parse
from medem.constants import DIR2DOCS_IMG, DIR2PAPER_IMG

import polars as pl


def to_pandas(
    df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    """Convert a polars dataframe to a pandas dataframe.

    Args:
        df (Union[pl.LazyFrame, pl.DataFrame]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect().to_pandas()
    elif isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        pass
    else:
        raise ValueError(
            f"df must be a polars dataframe or a pandas dataframe, got {type(df)} instead"
        )
    return df


def to_polars(
    df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]
) -> pl.DataFrame:
    """Convert a pandas dataframe to a polars dataframe.

    Args:
        df (Union[pl.LazyFrame, pl.DataFrame]): _description_

    Returns:
        pl.DataFrame: _description_
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    elif isinstance(df, pl.DataFrame):
        pass
    elif isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    else:
        raise ValueError(
            f"df must be a polars dataframe or a pandas dataframe, got {type(df)} instead"
        )
    return df


def to_lazyframe(
    df: Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]
) -> pl.LazyFrame:
    """Convert a pandas dataframe to a polars dataframe.

     Args:
        df (Union[pl.LazyFrame, pl.DataFrame, pd.DataFrame]): _description_

    Returns:
        pl.DataFrame: _description_
    """
    if isinstance(df, pl.LazyFrame):
        pass
    else:
        df = to_polars(df).lazy()
    return df


def save_figure_to_folders(
    fname_path: str,
    to_doc_dir: bool = True,
    to_paper_dir: bool = False,
    pdf=True,
):
    fname_path = Path(fname_path)
    reference_folder = fname_path.parents[0]
    if to_doc_dir:
        (DIR2DOCS_IMG / reference_folder).mkdir(exist_ok=True, parents=True)
        plt.savefig(
            str(DIR2DOCS_IMG / reference_folder / f"{fname_path.name}.png"),
            bbox_inches="tight",
        )
        if pdf:
            plt.savefig(
                str(DIR2DOCS_IMG / reference_folder / f"{fname_path.name}.pdf"),
                bbox_inches="tight",
            )

    if to_paper_dir:
        if not DIR2PAPER_IMG.exists():
            raise ValueError(f"{DIR2PAPER_IMG} does not exists, skipping")
        else:
            (DIR2PAPER_IMG / reference_folder).mkdir(
                exist_ok=True, parents=True
            )
            plt.savefig(
                str(
                    DIR2PAPER_IMG / reference_folder / f"{fname_path.name}.png"
                ),
                bbox_inches="tight",
            )
            if pdf:
                plt.savefig(
                    str(
                        DIR2PAPER_IMG
                        / reference_folder
                        / f"{fname_path.name}.pdf"
                    ),
                    bbox_inches="tight",
                )


def force_datetime(
    df: pd.DataFrame,
    colnames_datetime: Union[str, List[str]] = ["start"],
):
    """Force the conversion of the given columns to datetime.

    Args:
        df (pd.DataFrame): _description_
        colnames_datetime (Union[str, List[str]], optional): _description_. Defaults to ["start"].
        framework_ (pd or ks, optional): Decide which backend to use.. Defaults to pd.

    Returns:
        _type_: _description_
    """
    if isinstance(colnames_datetime, str):
        colnames_datetime = [colnames_datetime]
    for col_ in colnames_datetime:
        df[col_] = pd.to_datetime(df[col_])
    return df


def add_age(
    df: pd.DataFrame,
    ref_datetime: Union[int, datetime, str],
    birth_datetime_col: str = "birth_datetime",
    colname_age: str = None,
) -> pd.DataFrame:
    """Add a new column with the age built with the respect to the given
    reference (year, datetime, datetime column of the dataframe).

    Args:
        df (pd.DataFrame): _description_ ref_datetime (Union[int, datetime,
        str]): _description_
        birth_datetime_col (str, optional): _description_. Defaults to "birth_datetime".
        framework_ (Union[pd, ks], optional):  decide which backend to use.
    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_ = force_datetime(
        df,
        colnames_datetime=birth_datetime_col,
    )

    if type(ref_datetime) == int:
        ref_datetime_ = parse(f"{ref_datetime}-01-01 12:00:00")
    elif type(ref_datetime) == datetime:
        ref_datetime_ = ref_datetime
    elif type(ref_datetime) == str:
        if ref_datetime not in df_.columns:
            raise ValueError(f"{ref_datetime} not a column of the dataframe.")
        ref_datetime_ = pd.to_datetime(df_[ref_datetime])
    # build the label for the new column
    if type(ref_datetime) in [int, datetime]:
        ref_label = f"in_{ref_datetime_.year}"
    else:
        ref_label = f"to_{ref_datetime}"

    df_["age_in_days"] = (ref_datetime_ - df_[birth_datetime_col]).dt.days
    if colname_age is None:
        colname_age = f"age_{ref_label}"
    else:
        df_[colname_age] = df_["age_in_days"] / 365
    return df_


def clean_date_cols(df: pl.DataFrame) -> pl.DataFrame:
    """The date cols have wrong type in the OMOP extractions. Drop these useless
    columns and use datetime instead.

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    date_cols = [col for col in df.columns if col.endswith("date")]
    if len(date_cols) > 0:
        return df.drop(date_cols)
    else:
        return df


def make_df(
    string: str,
    autoconvert_datetimes: bool = True,
    **kwargs,
):
    buffer = io.StringIO(string)
    df = pd.read_csv(buffer, skipinitialspace=True, **kwargs)

    if autoconvert_datetimes:
        for col, dtype in df.dtypes.items():
            if dtype == "object":
                if df[col].str.match(r"\d{4}-\d{2}-\d{2}.*").any():
                    df[col] = pd.to_datetime(df[col])
    return df


# hdfs stuff for eds


def get_command_result(
    command: Union[List[str], str], as_shell: bool = False
) -> str:
    """
    Execute a shell command with subprocess and return the result.

    Parameters
    ----------
    command : str
        Command to execute.
    as_shell : str
        Wether to set `shell=True` or not to `check_output`

    Returns
    -------
    str
        Result.
    """

    if isinstance(command, str) and not as_shell:
        command = command.split()

    return subprocess.check_output(
        command, stderr=subprocess.DEVNULL, shell=as_shell
    ).decode()


def activate_hdfs():
    os.environ["ARROW_LIBHDFS_DIR"] = "/usr/local/hadoop/usr/lib/"
    os.environ["HADOOP_HOME"] = "/usr/local/hadoop"
    if Path(os.environ["HADOOP_HOME"]).exists():
        logger.info("Activating HDFS")
        os.environ["CLASSPATH"] = get_command_result("hadoop classpath --glob")
