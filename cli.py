from lightning import pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from imputation_model import ImputationModel
from dataset import ImputationDataModule



def main():
    cli = LightningCLI(
        model_class=ImputationModel,
        datamodule_class=ImputationDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_defaults={
            "callbacks": [
                LearningRateMonitor(logging_interval="step"),
            ],
        },
    )



if __name__ == "__main__":
    main()