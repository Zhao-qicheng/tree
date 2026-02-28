"""
数据集划分脚本 (split_dataset.py)

用于将原始 NPY 数据集划分为训练集和测试集。
支持两种模式：
1. `--mode random`：将所有文件打乱，按指定比例（如 8:2）拆分。
2. `--mode skater`：按运动员名称划分（例如将 Skater_D 划入测试集，其余在训练集）。
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List

import config

def get_all_npy_files(data_dir: str) -> List[Path]:
    """递归获取目录下所有的 .npy 文件"""
    return list(Path(data_dir).rglob("*.npy"))

def copy_file_to_dest(src_file: Path, base_src_dir: Path, dest_dir: Path) -> None:
    """
    将文件复制到目标目录，并保持其在原始目录下的相对层级结构。
    例如 src: data/npy/Skater_A/Axel/1.npy, base: data/npy
    dest: data/npy_train/Skater_A/Axel/1.npy
    """
    # 计算相对路径
    rel_path = src_file.relative_to(base_src_dir)
    dest_file = dest_dir / rel_path
    
    # 确保目标目录存在
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    shutil.copy2(src_file, dest_file)

def split_random(source_dir: Path, train_dir: Path, test_dir: Path, train_ratio: float, seed: int):
    """随机比例划分"""
    print(f"模式: 随机划分 (训练比例: {train_ratio:.2f}, 随机种子: {seed})")
    files = get_all_npy_files(source_dir)
    if not files:
        print(f"错误：在 {source_dir} 未找到任何 .npy 文件！")
        return

    random.seed(seed)
    random.shuffle(files)

    train_count = int(len(files) * train_ratio)
    train_files = files[:train_count]
    test_files = files[train_count:]

    print(f"总文件数: {len(files)} | 训练集: {len(train_files)} | 测试集: {len(test_files)}")

    print("正在复制训练集...")
    for f in train_files:
        copy_file_to_dest(f, source_dir, train_dir)
        
    print("正在复制测试集...")
    for f in test_files:
        copy_file_to_dest(f, source_dir, test_dir)

def split_by_skater(source_dir: Path, train_dir: Path, test_dir: Path, test_skaters: List[str]):
    """按运动员(Skater)划分，指定的名字划分到测试集，其余在训练集"""
    print(f"模式: 按运动员划分 (测试集 Skaters: {test_skaters})")
    files = get_all_npy_files(source_dir)
    if not files:
        print(f"错误：在 {source_dir} 未找到任何 .npy 文件！")
        return

    train_files = []
    test_files = []

    for f in files:
        # 假设层级为: data/npy/Skater_A/Axel/...
        # f.relative_to(source_dir).parts[0] 即为 "Skater_A"
        relative_parts = f.relative_to(source_dir).parts
        if not relative_parts:
            train_files.append(f)
            continue
            
        skater_name = relative_parts[0]
        
        if skater_name in test_skaters:
            test_files.append(f)
        else:
            train_files.append(f)

    print(f"总文件数: {len(files)} | 训练集: {len(train_files)} | 测试集: {len(test_files)}")

    print("正在复制训练集...")
    for f in train_files:
        copy_file_to_dest(f, source_dir, train_dir)
        
    print("正在复制测试集...")
    for f in test_files:
        copy_file_to_dest(f, source_dir, test_dir)

def main():
    parser = argparse.ArgumentParser(description="划分八叉树 NPY 数据集为训练集和测试集")
    parser.add_argument("--source", type=str, default=getattr(config, "FS_JUMP3D_DATA_DIR", ""), 
                        help="原始数据集路径 (默认为 config.FS_JUMP3D_DATA_DIR)")
    parser.add_argument("--train_dest", type=str, default=getattr(config, "TRAIN_DATA_DIR", "data/npy_train"),
                        help="训练集输出路径 (默认为 config.TRAIN_DATA_DIR)")
    parser.add_argument("--test_dest", type=str, default=getattr(config, "TEST_DATA_DIR", "data/npy_test"),
                        help="测试集输出路径 (默认为 config.TEST_DATA_DIR)")
    
    # 模式选择: random 或 skater
    parser.add_argument("--mode", type=str, choices=["random", "skater"], default="skater",
                        help="划分模式: 'random' (随机比例) 或 'skater' (按运动员)")
    
    # 随机模式参数
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="随机划分时的训练集比例，例 0.8 表示 80%% 训练，20%% 测试 (默认: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机数种子 (默认: 42)")
    
    # Skater模式参数
    parser.add_argument("--test_skaters", type=str, nargs='+', default=["Skater_D"],
                        help="当使用 skater 模式时，指定放入测试集的运动员文件夹名称 (例如: Skater_D Skater_E)")
    
    parser.add_argument("--force", action="store_true", default="--force",help="如果输出目录存在，强制覆盖(先清除原目录)")

    args = parser.parse_args()

    source_dir = Path(args.source)
    train_dir = Path(args.train_dest)
    test_dir = Path(args.test_dest)

    if not source_dir.exists() or not source_dir.is_dir():
        print(f"致命错误：原始数据目录 '{source_dir}' 不存在！")
        print("请检查路径或在 config.py 中正确配置 FS_JUMP3D_DATA_DIR")
        return

    # 处理输出目录重置
    for d in [train_dir, test_dir]:
        if d.exists():
            if args.force:
                print(f"清理已存在的目录: {d}")
                shutil.rmtree(d)
            else:
                print(f"致命错误：目标目录 '{d}' 已存在。为了防止数据混乱，请先手动删除，或使用 --force 参数。")
                return

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("开始划分数据集")
    print(f"数据源: {source_dir}")
    print(f"训练集: {train_dir}")
    print(f"测试集: {test_dir}")
    print("-" * 60)

    if args.mode == "random":
        split_random(source_dir, train_dir, test_dir, args.ratio, args.seed)
    elif args.mode == "skater":
        split_by_skater(source_dir, train_dir, test_dir, args.test_skaters)

    print("-" * 60)
    print("✅ 数据集划分完成！")
    print("请紧接着修改 config.py：")
    print(f"  TRAIN_DATA_DIR = '{train_dir.absolute().as_posix()}'")
    print(f"  TEST_DATA_DIR  = '{test_dir.absolute().as_posix()}'")
    print("然后运行 train.py 进行训练。")
    print("=" * 60)

if __name__ == "__main__":
    main()
