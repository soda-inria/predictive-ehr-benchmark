# Usage

## Pre-requisites

### Data and compute environment

This code is focused on the computing and data environment from Paris Greater Hospitals. The data format should be OMOP.

### Install the package

Install poetry and python 3.10. Inside the project folder, run the following command:

```bash
poetry install
```

## Create the study population

Three populations are available, corresponding to three predictive tasks: length of stay interpolation (LOS), prognosis of the next diagnosis billing codes (grouped into 21 ICD10 chapters), prognosis of major adverse cardio-vascular events (MACE). 

- The scripts for building each population are in `medem.populations`.To build the length-of-stay population, run: 

```bash
poetry run python medem.populations.t1_los_population.py
```  

- The configuration for each populations are in `medem.exeperiences.configurations.py`. Most importantly, the user specifies the database name. At loading time, the code will look for a database inside the hive database of the APHP at: `hdfs://bbsedsi/apps/hive/warehouse/bigdata/omop_exports_prod/hive/{database_name}.db/`. All parameters that can be specified are:

```
 "database_name": "cse210038_20220921_160214312112",
    "cohort_name": "complete_hospitalization_los", # name 
    "study_start": parse("2017-01-01"), # start of the study period
    "study_end": parse("2022-06-01"), # end of the study period. Outside of this range, all data is thrown away 
    "min_age_at_admission": 18, # minimum age at admission
    "sup_quantile_visits": 0.95, # exclude patients having a number of visits above the resulting threshold number of visits per patient
    "task_name": TASK_LOS_CATEGORICAL, # define the prognosis task 
    "los_categories": np.array([0, 7, np.inf]), # define the categories for the task,
    "with_incomplete_hospitalization": True, # include also outpatient visits ? 
    "visits_w_billing_codes_only": True, # keep only visits with billing codes ?
    "horizon_in_days": 30,  # used for avoiding right-censoring
    "event_tables": DEFAULT_EVENT_CONFIG, # what event tables to use as features
    "test_size": 0.2,
    "n_min_events": 10, # minimum number of events for a given medical code be included. Too rare events are thrown away.
```

- The underlying functions used for all population are `medem.preprocessing.selection.py:select_population` and `medem.preprocessing.selection.py:create_outcome`. Refer to these functions to see every details on the population flowcharts and task definitions.


## Run an experiment

- The scripts to benchmark different machine learning pipelines are in `medem.experiences.setups`. For
  example to launch the benchmark on the :

```bash
poetry run python medem.experiments.setups/los_prediction.py
```

- The configurations for the experiments are in `medem.experiments.configurations.py`. The available parameters are: 

```python
CONFIG_LOS_ESTIMATION = {
    "validation_size": 0.1,  # validation size
    "subtrain_size": [0.1, 0.5, 1] # size of the succesives effective train sets 
    "splitting_rs": list(range(5)),  # random seeds 
    "estimator_config": ESTIMATORS_TASK_LOS, # list of estimators to benchmark
    "featurizer_config": FEATURIZERS_TASK_LOS, # list of featurizers to benchmark
    "randomsearch_scoring": "roc_auc", # scoring function for the random search
    "randomsearch_n_iter": 10, # number of random search iterations
    "randomsearch_rs": 0, # random seed for the random search
    "n_min_events": 10, # minimum number of events for a given medical code be included. Too rare events are thrown away.
    "colname_demographics": [
        STATIC_FEATURES_DICT,
    ], # list of static features to include
    "local_embeddings_params": {
        "colname_concept": COLNAME_SOURCE_CODE,
        "window_radius_in_days": 30,
        "window_orientation": "center",
        "backend": "pandas",
        "d": N_COMPONENTS,
    }, # parameters passed to the local embeddings pipeline if present in featurizer config
}
```

- To use the slurm cluster on the AP-HP data warehouse, use the dedicated sbatch scripts. You might need to change some path in these scripts that are specific to a given AP-HP user and project: 

```
cd scripts/experiences/
mkdir logs
sbatch los_sbatch.sh
```

- NB: all codes to run experiment with the transformer model are in  [a fork from the original cehr-bert transformer model](https://github.com/strayMat/cehr-bert).