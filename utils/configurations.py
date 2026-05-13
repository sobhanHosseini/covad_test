from enum import Enum


class LearningType(str, Enum):
    ONE_CLASS = "one_class"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class Split(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class LabelName(int, Enum):
    NORMAL = 0
    ABNORMAL = 1
