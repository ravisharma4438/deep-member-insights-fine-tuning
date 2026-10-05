"""Launch a LoRA fine-tuning job on SageMaker (SDK v3 ModelTrainer). Run from Studio:

    python -m dmi.train.launch --model gemma-4-12b --dataset screens-20251205-20261005
    python -m dmi.train.launch --model gemma-4-12b --dataset <name> --smoke   # ~10 steps, no merge

The job uploads src/ (dmi package) and runs dmi/train/sft.py under torchrun on
a PyTorch training image, with dmi/train/requirements.txt installed on top.
Artifacts land uncompressed at
    s3://$DMI_S3_BUCKET/$DMI_S3_PREFIX/training-jobs/<job>/output/model/{adapter,merged}/
"""

from __future__ import annotations

import argparse

from dmi import settings
from dmi.train import models

PYTORCH_IMAGE_VERSION = "2.10.0"
PYTORCH_PY_VERSION = "py313"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(models.MODELS))
    parser.add_argument("--dataset", required=True, help="name under s3://.../datasets/")
    parser.add_argument("--instance-type", help="override the model's default")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--smoke", action="store_true", help="10 steps on 64 samples, no merge")
    parser.add_argument("--max-runtime-hours", type=float, default=48)
    parser.add_argument("--wait", action="store_true", help="stream logs until the job finishes")
    args = parser.parse_args()

    from sagemaker.core import image_uris
    from sagemaker.core.helper.session_helper import Session, get_execution_role
    from sagemaker.train.distributed import Torchrun
    from sagemaker.train.model_trainer import ModelTrainer

    try:
        from sagemaker.core.training.configs import (
            Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition,
        )
    except ImportError:  # older 3.x layouts
        from sagemaker.train.configs import (
            Compute, InputData, OutputDataConfig, SourceCode, StoppingCondition,
        )

    spec = models.get(args.model)
    instance_type = args.instance_type or spec.instance_type
    session = Session()
    role = settings.get("SAGEMAKER_ROLE_ARN") or get_execution_role(session)

    image = image_uris.retrieve(
        framework="pytorch",
        region=session.boto_region_name,
        version=PYTORCH_IMAGE_VERSION,
        py_version=PYTORCH_PY_VERSION,
        instance_type=instance_type,
        image_scope="training",
    )

    hyperparameters: dict[str, object] = {"model_key": spec.key}
    for key, value in {
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
    }.items():
        if value is not None:
            hyperparameters[key] = value
    if args.smoke:
        hyperparameters.update(max_steps=10, max_train_samples=64, max_eval_samples=8, merge=False)

    environment = {"TOKENIZERS_PARALLELISM": "false"}
    hf_token = settings.get("HF_TOKEN")
    if hf_token:
        environment["HF_TOKEN"] = hf_token  # gated models (Gemma, Llama)

    trainer = ModelTrainer(
        training_image=image,
        role=role,
        sagemaker_session=session,
        base_job_name=f"dmi-{spec.key}{'-smoke' if args.smoke else ''}",
        source_code=SourceCode(
            source_dir=str(settings.REPO_ROOT / "src"),
            entry_script="dmi/train/sft.py",
            requirements="dmi/train/requirements.txt",
        ),
        distributed=Torchrun(process_count_per_node=spec.gpus_per_instance),
        compute=Compute(
            instance_type=instance_type,
            instance_count=1,
            volume_size_in_gb=spec.volume_size_gb,
        ),
        stopping_condition=StoppingCondition(max_runtime_in_seconds=int(args.max_runtime_hours * 3600)),
        output_data_config=OutputDataConfig(
            s3_output_path=settings.s3_uri("training-jobs"),
            compression_type="NONE",
        ),
        hyperparameters=hyperparameters,
        environment=environment,
    )

    dataset_uri = settings.s3_uri("datasets", args.dataset)
    print(f"Launching {spec.key} ({spec.hf_id}) on {instance_type} with {dataset_uri}")
    print(f"Image: {image}")
    trainer.train(
        input_data_config=[InputData(channel_name="training", data_source=dataset_uri + "/")],
        wait=args.wait,
    )
    job = trainer._latest_training_job
    name = job.training_job_name if job else "(see SageMaker console)"
    print(f"Training job: {name}")
    print(f"Artifacts:    {settings.s3_uri('training-jobs', name, 'output', 'model')}/")


if __name__ == "__main__":
    main()
