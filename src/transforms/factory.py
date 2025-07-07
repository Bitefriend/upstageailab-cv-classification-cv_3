from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    input_size: tuple[int, int]
    mean: tuple[int, int, int]
    std: tuple[int, int, int]


def get_model_config(model_name: str) -> ModelConfig:
    import timm

    model = timm.create_model(model_name, pretrained=True)
    input_shape = model.default_cfg["input_size"]  # (channel, width, height)
    return ModelConfig(
        input_size=(input_shape[1], input_shape[2]),
        mean=model.default_cfg["mean"],
        std=model.default_cfg["std"],
    )


def create_train_test_transforms(model_name: str) -> tuple[Callable, Callable]:
    """사전 학습된 model 의 설정에 맞추고 online augmentation 을 포함한 train transform 과 test transform 을 만들어 줌"""
    from torchvision import transforms

    model_config = get_model_config(model_name)

    train_transform = transforms.Compose(
        [
            transforms.Resize(model_config.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.mean, std=model_config.std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(model_config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.mean, std=model_config.std),
        ]
    )
    return train_transform, test_transform
