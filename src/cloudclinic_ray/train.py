import argparse
import logging
import os
import socket
from collections.abc import Iterable
from io import BytesIO

import ray
import torch
from PIL import Image
from ray import data as ray_data
from ray import train as ray_train
from ray.train import torch as ray_torch
from torch import nn
from torchvision.transforms import ToTensor

WORKER_BATCH_SIZE = 128

DATASET_NAME = "zalando-datasets/fashion_mnist"
DATASET_REVISION = "531be5e2ccc9dba0c201ad3ae567a4f3d16ecdd2"
SOURCE_URLS = {
    split: (
        f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/{DATASET_REVISION}"
        f"/fashion_mnist/{split}-00000-of-00001.parquet"
    )
    for split in ("train", "test")
}


class MLP(nn.Module):
    """A small multilayer perceptron for 28x28 images and 10 output classes.

    📘 Sourced from https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.flatten(x))


def evaluate(
    model: nn.Module, batches: Iterable, loss_fn: nn.Module
) -> tuple[float, float]:
    """Compute average loss and accuracy."""
    total, correct, seen = 0.0, 0, 0

    # ❗Put model in evaluation mode and disable gradient calculation.
    model.eval()
    with torch.no_grad():
        for batch in batches:
            x, y = batch["image"], batch["label"]
            logits = model(x)
            total += loss_fn(logits, y).item() * y.shape[0]
            correct += int((logits.argmax(1) == y).sum().item())
            seen += y.shape[0]

    return total / max(seen, 1), correct / max(seen, 1)


def decode_image(row: dict) -> dict:
    """Convert image bytes to grayscale pixels scaled to [0, 1]."""
    with Image.open(BytesIO(row["image"]["bytes"])) as image:
        pixels = ToTensor()(image.convert("L")).numpy()

    return {"image": pixels, "label": int(row["label"])}


def get_datasets() -> dict:
    """Decode and cache Fashion-MNIST dataset."""
    data_context = ray_data.DataContext.get_current()
    data_context.enable_progress_bars = False
    data_context.execution_options.preserve_order = True

    return {
        split: ray_data.read_parquet(url).map(decode_image).materialize()
        for split, url in SOURCE_URLS.items()
    }


def train(config: dict) -> None:
    """Train and evaluate the model on each Ray worker."""

    # Identify this worker for logging and to set a rank-specific seed.
    context = ray_train.get_context()
    rank, world_size = context.get_world_rank(), context.get_world_size()
    print(
        f"Worker {rank}/{world_size} on {socket.gethostname()} "
        f"using {ray_torch.get_device()}"
    )

    # Ray Data shards the data across the all workers.
    # Here we're fetching the shard assigned to this worker.
    train_data = ray_train.get_dataset_shard("train")
    test_data = ray_train.get_dataset_shard("test")
    assert train_data is not None and test_data is not None

    torch.manual_seed(config["seed"] + rank)

    # Each worker trains a model copy on its own data shard. Ray uses
    # DistributedDataParallel (DDP) to synchronize gradients across workers.
    # 📘 https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
    model = ray_torch.prepare_model(MLP())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        # Ray Data creates tensor batches on this worker's CPU or GPU.
        train_batches = train_data.iter_torch_batches(batch_size=WORKER_BATCH_SIZE)
        test_batches = test_data.iter_torch_batches(batch_size=WORKER_BATCH_SIZE)

        for batch in train_batches:
            x, y = batch["image"], batch["label"]
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            # DDP synchronizes gradients across workers during backpropagation.
            loss.backward()
            optimizer.step()

        loss, accuracy = evaluate(model, test_batches, loss_fn)
        if rank == 0:
            print(f"Epoch {epoch + 1}: test_loss={loss:.4f}, accuracy={accuracy:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--gpu", action="store_true", help="Require one GPU per worker")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )

    return parser.parse_args()


# Based on https://docs.skypilot.ai/en/stable/examples/training/ray.html
def main():
    args = _parse_args()
    logging.getLogger("ray.data").setLevel("WARNING")

    # ❗Connect to the cluster at RAY_ADDRESS or start Ray locally.
    with ray.init(
        address=os.getenv("RAY_ADDRESS", "local"),
        runtime_env={"env_vars": {"RAY_DATA_LOG_LEVEL": "WARNING"}},
    ):
        if args.gpu and ray.cluster_resources().get("GPU", 0) < args.n_workers:
            msg = "Ray needs one GPU per worker"
            raise ValueError(msg)

        scaling_config = ray_train.ScalingConfig(
            num_workers=args.n_workers, use_gpu=args.gpu
        )
        # Run `train()` on each worker with the same configuration.
        ray_torch.TorchTrainer(
            train_loop_per_worker=train,
            train_loop_config={"epochs": args.epochs, "seed": 9152016},
            scaling_config=scaling_config,  # pyrefly: ignore
            datasets=get_datasets(),
            # ❗Split training data, but each worker gets the full test set.
            dataset_config=ray_train.DataConfig(datasets_to_split=["train"]),
        ).fit()


if __name__ == "__main__":
    main()
