import os

import pandas as pd
from databricks import koalas as ks
from eds_scikit import improve_performances
from eds_scikit.io import HiveData
from edstoolbox import SparkApp
from event2vec.svd_ppmi import event2vec
from loguru import logger
from pyspark.sql import SparkSession

from medem.constants import (
    COLNAME_SOURCE_CODE,
    COLNAME_SOURCE_TYPE,
    DIR2DATA,
    DIR2EMBEDDINGS,
)
from medem.experiences.configurations import (
    CONFIG_LOS_COHORT,
    cohort_configuration_to_str,
)
from medem.experiences.utils import build_vocabulary, restrict_to_vocabulary

app = SparkApp("event2vec")


@app.submit
def run(spark, sql, config):
    """
    Submit event2vec on APHP pyspark cluster for precomputed events:
    - cd to medem/scripts/experiences (or where this file resids)
    - run `s-toolbox spark submit --config scripts/experiences/config.cfg --log-path logs scripts/experiences/build_event2vec_from_large_cohort.py`

    Warning: the local file system is not accessible for write access for the client, so the hdfs file system will be used (supposingly faster)

    More documentation at : https://datasciencetools-pages.eds.aphp.fr/edstoolbox/cli/spark/

    Arguments
    ---------
    spark :
        Spark session object, for querying tables
    sql :
        Spark sql object, for querying tables
    config :
        Dictionary containing the configuration
    """

    #
    user = os.getenv("USER")
    # hdfs_path = f"hdfs://bbsedsi/user/{user}/"+"anti_"+cohort_name + "/event.parquet"

    anti_cohort_name = "anti_" + cohort_configuration_to_str(CONFIG_LOS_COHORT)
    logger.info(
        f"Embeddings will be saved at hdfs://bbsedsi/user/{user}/{anti_cohort_name}"
    )
    anti_event = pd.read_parquet(
        DIR2DATA / anti_cohort_name / "event.parquet",
        columns=[
            "person_id",
            "start",
            COLNAME_SOURCE_CODE,
            COLNAME_SOURCE_TYPE,
        ],
    )
    logger.info(anti_event.columns)
    logger.info(anti_event[COLNAME_SOURCE_TYPE].value_counts())

    n_min_events = 10
    vocabulary = build_vocabulary(event=anti_event, n_min_events=n_min_events)
    logger.info(f"vocabulary size: {len(vocabulary)}")
    restricted_anti_event = restrict_to_vocabulary(
        event=anti_event, vocabulary=vocabulary
    )
    logger.info(restricted_anti_event[COLNAME_SOURCE_TYPE].value_counts())
    matrix_type = "numpy"

    if matrix_type == "numpy":
        dir2output = DIR2EMBEDDINGS / anti_cohort_name
    elif matrix_type == "parquet":
        dir2output = anti_cohort_name
    else:
        raise ValueError("wrong matrix type")
    embeddings = event2vec(
        events=spark.createDataFrame(restricted_anti_event),
        output_dir=dir2output,
        colname_concept=COLNAME_SOURCE_CODE,
        window_radius_in_days=2 * CONFIG_LOS_COHORT["horizon_in_days"],
        window_orientation="center",
        matrix_type=matrix_type,
        backend="spark",
        d=30,
        smoothing_factor=0.75,
        k=1,
    )


if __name__ == "__main__":
    app.run()
