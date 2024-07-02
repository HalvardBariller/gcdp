"This script is used to configure the logging for the project."

# The following code was adapted from the project LeRobot:
# https://github.com/huggingface/lerobot

import logging
import os
import sys

from datetime import datetime


def init_logging():
    def custom_format(record):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fnameline = f"{record.pathname}:{record.lineno}"
        message = f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.getMessage()}"
        return message

    logging.basicConfig(level=logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter()
    formatter.format = custom_format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


# class Logger:
#     """Primary logger object. Logs either locally or using wandb.

#     The logger creates the following directory structure:

#     provided_log_dir
#     ├── .hydra  # hydra's configuration cache
#     ├── checkpoints
#     │   ├── specific_checkpoint_name
#     │   │   ├── pretrained_model  # Hugging Face pretrained model directory
#     │   │   │   ├── ...
#     │   │   └── training_state.pth  # optimizer, scheduler, and random states + training step
#     |   ├── another_specific_checkpoint_name
#     │   │   ├── ...
#     |   ├── ...
#     │   └── last  # a softlink to the last logged checkpoint
#     """

#     pretrained_model_dir_name = "pretrained_model"
#     training_state_file_name = "training_state.pth"

#     def __init__(self, cfg: DictConfig, log_dir: str, wandb_job_name: str | None = None):
#         """
#         Args:
#             log_dir: The directory to save all logs and training outputs to.
#             job_name: The WandB job name.
#         """
#         self._cfg = cfg
#         self.log_dir = Path(log_dir)
#         self.log_dir.mkdir(parents=True, exist_ok=True)
#         self.checkpoints_dir = self.get_checkpoints_dir(log_dir)
#         self.last_checkpoint_dir = self.get_last_checkpoint_dir(log_dir)
#         self.last_pretrained_model_dir = self.get_last_pretrained_model_dir(log_dir)

#         # Set up WandB.
#         self._group = cfg_to_group(cfg)
#         project = cfg.get("wandb", {}).get("project")
#         entity = cfg.get("wandb", {}).get("entity")
#         enable_wandb = cfg.get("wandb", {}).get("enable", False)
#         run_offline = not enable_wandb or not project
#         if run_offline:
#             logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))
#             self._wandb = None
#         else:
#             os.environ["WANDB_SILENT"] = "true"
#             import wandb

#             wandb_run_id = None
#             if cfg.resume:
#                 wandb_run_id = get_wandb_run_id_from_filesystem(self.checkpoints_dir)

#             wandb.init(
#                 id=wandb_run_id,
#                 project=project,
#                 entity=entity,
#                 name=wandb_job_name,
#                 notes=cfg.get("wandb", {}).get("notes"),
#                 tags=cfg_to_group(cfg, return_list=True),
#                 dir=log_dir,
#                 config=OmegaConf.to_container(cfg, resolve=True),
#                 # TODO(rcadene): try set to True
#                 save_code=False,
#                 # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
#                 job_type="train_eval",
#                 resume="must" if cfg.resume else None,
#             )
#             print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
#             logging.info(f"Track this run --> {colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}")
#             self._wandb = wandb

    # @classmethod
    # def get_checkpoints_dir(cls, log_dir: str | Path) -> Path:
    #     """Given the log directory, get the sub-directory in which checkpoints will be saved."""
    #     return Path(log_dir) / "checkpoints"

    # @classmethod
    # def get_last_checkpoint_dir(cls, log_dir: str | Path) -> Path:
    #     """Given the log directory, get the sub-directory in which the last checkpoint will be saved."""
    #     return cls.get_checkpoints_dir(log_dir) / "last"

    # @classmethod
    # def get_last_pretrained_model_dir(cls, log_dir: str | Path) -> Path:
    #     """
    #     Given the log directory, get the sub-directory in which the last checkpoint's pretrained weights will
    #     be saved.
    #     """
    #     return cls.get_last_checkpoint_dir(log_dir) / cls.pretrained_model_dir_name

    # def save_model(self, save_dir: Path, policy: Policy, wandb_artifact_name: str | None = None):
    #     """Save the weights of the Policy model using PyTorchModelHubMixin.

    #     The weights are saved in a folder called "pretrained_model" under the checkpoint directory.

    #     Optionally also upload the model to WandB.
    #     """
    #     self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    #     policy.save_pretrained(save_dir)
    #     # Also save the full Hydra config for the env configuration.
    #     OmegaConf.save(self._cfg, save_dir / "config.yaml")
    #     if self._wandb and not self._cfg.wandb.disable_artifact:
    #         # note wandb artifact does not accept ":" or "/" in its name
    #         artifact = self._wandb.Artifact(wandb_artifact_name, type="model")
    #         artifact.add_file(save_dir / SAFETENSORS_SINGLE_FILE)
    #         self._wandb.log_artifact(artifact)
    #     if self.last_checkpoint_dir.exists():
    #         os.remove(self.last_checkpoint_dir)

    # def save_training_state(
    #     self,
    #     save_dir: Path,
    #     train_step: int,
    #     optimizer: Optimizer,
    #     scheduler: LRScheduler | None,
    # ):
    #     """Checkpoint the global training_step, optimizer state, scheduler state, and random state.

    #     All of these are saved as "training_state.pth" under the checkpoint directory.
    #     """
    #     training_state = {
    #         "step": train_step,
    #         "optimizer": optimizer.state_dict(),
    #         **get_global_random_state(),
    #     }
    #     if scheduler is not None:
    #         training_state["scheduler"] = scheduler.state_dict()
    #     torch.save(training_state, save_dir / self.training_state_file_name)

    # def save_checkpont(
    #     self,
    #     train_step: int,
    #     policy: Policy,
    #     optimizer: Optimizer,
    #     scheduler: LRScheduler | None,
    #     identifier: str,
    # ):
    #     """Checkpoint the model weights and the training state."""
    #     checkpoint_dir = self.checkpoints_dir / str(identifier)
    #     wandb_artifact_name = (
    #         None
    #         if self._wandb is None
    #         else f"{self._group.replace(':', '_').replace('/', '_')}-{self._cfg.seed}-{identifier}"
    #     )
    #     self.save_model(
    #         checkpoint_dir / self.pretrained_model_dir_name, policy, wandb_artifact_name=wandb_artifact_name
    #     )
    #     self.save_training_state(checkpoint_dir, train_step, optimizer, scheduler)
    #     os.symlink(checkpoint_dir.absolute(), self.last_checkpoint_dir)

    # def load_last_training_state(self, optimizer: Optimizer, scheduler: LRScheduler | None) -> int:
    #     """
    #     Given the last checkpoint in the logging directory, load the optimizer state, scheduler state, and
    #     random state, and return the global training step.
    #     """
    #     training_state = torch.load(self.last_checkpoint_dir / self.training_state_file_name)
    #     optimizer.load_state_dict(training_state["optimizer"])
    #     if scheduler is not None:
    #         scheduler.load_state_dict(training_state["scheduler"])
    #     elif "scheduler" in training_state:
    #         raise ValueError(
    #             "The checkpoint contains a scheduler state_dict, but no LRScheduler was provided."
    #         )
    #     # Small hack to get the expected keys: use `get_global_random_state`.
    #     set_global_random_state({k: training_state[k] for k in get_global_random_state()})
    #     return training_state["step"]

    # def log_dict(self, d, step, mode="train"):
    #     assert mode in {"train", "eval"}
    #     # TODO(alexander-soare): Add local text log.
    #     if self._wandb is not None:
    #         for k, v in d.items():
    #             if not isinstance(v, (int, float, str)):
    #                 logging.warning(
    #                     f'WandB logging of key "{k}" was ignored as its type is not handled by this wrapper.'
    #                 )
    #                 continue
    #             self._wandb.log({f"{mode}/{k}": v}, step=step)

    # def log_video(self, video_path: str, step: int, mode: str = "train"):
    #     assert mode in {"train", "eval"}
    #     assert self._wandb is not None
    #     wandb_video = self._wandb.Video(video_path, fps=self._cfg.fps, format="mp4")
    #     self._wandb.log({f"{mode}/video": wandb_video}, step=step)



# class Logger:
#     """Primary logger object. Logs either locally or using wandb."""

#     def __init__(self, cfg, log_dir, wandb_job_name=None):



#     def log_train_wandb(cfg, step, **kwargs):
#         """Log training metrics to WandB."""
        