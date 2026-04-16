"""
Run all training/evaluation jobs for all datasets and variants.

This is a convenience script that runs:
  - MRPC: base (eval) + tuned (train)
  - STS-B: base (eval) + tuned (train)
  - QQP: base (eval) + tuned (train)
"""

from training.mrpc import train_and_export_mrpc
from training.sts import train_and_export_sts
from training.qqp import run_qqp_benchmark


def main() -> None:
    print("=" * 60)
    print("RUNNING ALL DATASETS — BASE + TUNED")
    print("=" * 60)

    print("\n\n" + "=" * 60)
    print("MRPC")
    print("=" * 60)
    train_and_export_mrpc()

    print("\n\n" + "=" * 60)
    print("STS-B")
    print("=" * 60)
    train_and_export_sts()

    print("\n\n" + "=" * 60)
    print("QQP")
    print("=" * 60)
    run_qqp_benchmark()

    print("\n\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
