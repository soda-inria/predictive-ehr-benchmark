"""
Load a cohort and create the events as separate domain tables for cehr-bert
model. This is done to create a folder for pre-training.
"""

import argparse
from pathlib import Path
from typing import Dict
import pandas as pd
import polars as pl

from medem.constants import (
    DIR2DATA,
    COLNAME_PERSON,
)
from medem.preprocessing.utils import PolarsData
from medem.experiences.cohort import EventCohort
from medem.experiences.configurations import (
    COHORT_NAME2CONFIG,
    CONFIG_PROGNOSIS_COHORT,
    cohort_configuration_to_str,
    DEFAULT_EVENT_CONFIG,
)
from loguru import logger


# Necessary mappings
MAPPING_VISIT_CONCEPT_ID = {
    "consultation externe": 9202,
    "hospitalisation incomplète": 9201,
    "urgence": 9203,
    "hospitalisés": 9201,
}

OMOP_CONCEPT_MAPPINGS = {
    "visit_occurrence": {
        "source_column": "visit_source_value",
        "target_column": "visit_concept_id",
        "mapping": MAPPING_VISIT_CONCEPT_ID,
    }
}


def map_concept(table: pl.LazyFrame, table_name: str) -> pl.LazyFrame:
    mapping_dict = OMOP_CONCEPT_MAPPINGS[table_name]
    table_ = table.drop(mapping_dict["target_column"])
    mapping_df = pl.from_pandas(
        pd.DataFrame(
            mapping_dict["mapping"].items(),
            columns=[
                mapping_dict["source_column"],
                mapping_dict["target_column"],
            ],
        )
    ).lazy()
    return table_.join(mapping_df, on=mapping_dict["source_column"], how="left")


def convert_data_for_cerh_bert(cohort_config: Dict):
    database_name = cohort_config["database_name"]
    database = PolarsData(database_name=database_name)

    cohort_name = cohort_configuration_to_str(cohort_config)
    dir2cohort = DIR2DATA / cohort_name
    event_cohort = EventCohort(folder=dir2cohort)
    logger.info(event_cohort.person.shape)
    logger.info(event_cohort.event.shape)

    train_data_id = (
        pl.scan_parquet(dir2cohort / "dataset_split.parquet")
        .filter(pl.col("dataset") == "train")
        .select(COLNAME_PERSON)
    )

    cehr_bert_train_dir = dir2cohort / "cehr_bert_train"
    cehr_bert_train_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Number of events in cohort: {event_cohort.event.shape}")

    # TODO: This code does not scale for out of memory table events.
    # 1- try  full lazy and sink parquet ?
    # 2- Launch with extended memory in slurm with an sbatch

    for domain_table_name in [
        "person",
        *DEFAULT_EVENT_CONFIG.keys(),
        "visit_occurrence",
    ]:
        table_ = database.__getattribute__(domain_table_name).join(
            train_data_id, on=COLNAME_PERSON, how="inner"
        )
        # if necessary add OMOP standard concept ids
        if domain_table_name in OMOP_CONCEPT_MAPPINGS.keys():
            table_ = map_concept(table=table_, table_name=domain_table_name)
        # rename to be consistent with cert_behrt naming
        if domain_table_name == "drug_exposure_administration":
            domain_table_name = "drug_exposure"
        # need to convert datetime to str for pyspark processing....
        for col in table_.columns:
            if col.find("datetime") != -1:
                table_ = table_.with_columns(
                    pl.col(col).dt.strftime("%Y-%m-%d %H:%M:%S")
                )
        logger.info(f"write table at {cehr_bert_train_dir / domain_table_name}")
        table_.collect().write_parquet(
            cehr_bert_train_dir / domain_table_name, compression="snappy"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort_name",
        type=str,
        help="Path to train results, default is package data directory.",
        choices=list(COHORT_NAME2CONFIG.keys()),
    )
    args = parser.parse_args()
    cohort_config = COHORT_NAME2CONFIG[args.cohort_name]
    convert_data_for_cerh_bert(
        cohort_config=cohort_config,
    )
