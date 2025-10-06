from .efficientdet import EfficientDet, HeadNet
from .bench import DetBenchTrain, DetBenchPredict
from .config import get_efficientdet_config
from .helpers import load_pretrained, load_checkpoint
import torch


def create_model(
        model_name, bench_task='', num_classes=None, pretrained=False,
        checkpoint_path='', checkpoint_ema=False, **kwargs):

    config = get_efficientdet_config(model_name)
    return create_model_from_config(
        config, bench_task=bench_task, num_classes=num_classes, pretrained=pretrained,
        checkpoint_path=checkpoint_path, checkpoint_ema=checkpoint_ema, **kwargs)


def create_model_from_config(
        config, bench_task='', num_classes=None, pretrained=False,
        checkpoint_path='', checkpoint_ema=False, **kwargs):

    pretrained_backbone = kwargs.pop('pretrained_backbone', True)
    if pretrained or checkpoint_path:
        pretrained_backbone = False  # no point in loading backbone weights

    # Config overrides, override some config values via kwargs.
    overrides = (
        'redundant_bias', 'label_smoothing', 'legacy_focal', 'jit_loss', 'soft_nms', 'max_det_per_image', 'image_size')
    for ov in overrides:
        value = kwargs.pop(ov, None)
        if value is not None:
            setattr(config, ov, value)

    labeler = kwargs.pop('bench_labeler', False)

    # create the base model
    model = EfficientDet(config, pretrained_backbone=pretrained_backbone, **kwargs)

    # pretrained weights are always spec'd for original config, load them before we change the model
    if pretrained:
        load_pretrained(model, config.url)

    # reset model head if num_classes doesn't match configs
    if num_classes is not None and num_classes != config.num_classes:
        print("Head changed. ")
        model.reset_head(num_classes=num_classes)

    # load an argument specified training checkpoint
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        ckpt_state = checkpoint.get('state_dict', checkpoint)

        # Only load parameters whose shapes match the current model. This preserves trained
        # classification head weights when compatible, and safely skips mismatches.
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in ckpt_state.items():
            if k in model_state and hasattr(v, 'shape') and hasattr(model_state[k], 'shape'):
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    skipped.append(k)
            else:
                # allow unexpected keys to be skipped
                skipped.append(k)

        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        if skipped:
            print(f"Skipped incompatible keys (shape mismatch or unexpected): {len(skipped)}")
        print(f"Loaded with missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")


    # wrap model in task specific training/prediction bench if set
    if bench_task == 'train':
        model = DetBenchTrain(model, create_labeler=labeler)
    elif bench_task == 'predict':
        model = DetBenchPredict(model)
    return model
