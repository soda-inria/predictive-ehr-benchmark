# depracated
from edstoolbox import SlurmApp
from medem.experiences.setups.los_prediction import run_experience as los_run

# from medem.experiences.setups.prognosis import run_experience as prognosis_run
from medem.experiences.setups.mace_prediction import (
    run_experience as mace_run,
)

# launch with: eds-toolbox slurm submit --config efficiency_slurm_config.cfg --log-path logs efficiency_submit.py
app = SlurmApp()


@app.submit
def run(config=None):
    # can overwrite experience configs
    if config is not None:
        expe_config = config["experience"]
    else:
        expe_config = config["experience"]
    run_experience(config=expe_config)


if __name__ == "__main__":
    app.run()
