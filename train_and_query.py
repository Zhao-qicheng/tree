"""
兼容入口：顺序执行训练与查询。

建议直接使用 train.py 和 query.py。
"""

from __future__ import annotations

from train import train
from query import query


def main() -> None:
    train()
    query()


if __name__ == "__main__":
    main()

