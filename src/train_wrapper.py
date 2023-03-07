from src.weighted_trainer import DetectionTrainer

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.yolo.utils import LOGGER, RANK,  yaml_load
from ultralytics.yolo.utils.checks import check_yaml


def train(yolo_model, **kwargs):
    """
    Trains the model on a given dataset.

    Args:
        **kwargs (Any): Any number of arguments representing the training configuration.
    """
    overrides = yolo_model.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get("cfg"):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs["cfg"]), append_filename=True)
    overrides["task"] = yolo_model.task
    overrides["mode"] = "train"
    if not overrides.get("data"):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get("resume"):
        overrides["resume"] = yolo_model.ckpt_path

    yolo_model.trainer = DetectionTrainer(overrides=overrides)
    if not overrides.get("resume"):  # manually set model only if not resuming
        yolo_model.trainer.model = yolo_model.trainer.get_model(weights=yolo_model.model if yolo_model.ckpt else None, cfg=yolo_model.model.yaml)
        yolo_model.model = yolo_model.trainer.model
    yolo_model.trainer.train()
    # update model and cfg after training
    if RANK in {0, -1}:
        yolo_model.model, _ = attempt_load_one_weight(str(yolo_model.trainer.best))
        yolo_model.overrides = yolo_model.model.args