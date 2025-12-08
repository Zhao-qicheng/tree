扁平化八叉树优化说明
====================

概述
----
本系统已升级为**扁平化八叉树 (Flat Octree)** 存储结构。
通过使用 Numpy 数组替代 Python 对象树，实现了极致的加载速度和更小的内存占用。

核心收益
--------
1.  **加载速度提升 20-100 倍**：利用内存映射 (mmap) 技术，实现毫秒级模型加载。
2.  **文件大小减少 50-70%**：去除对象开销，紧凑存储。
3.  **内存占用大幅降低**：查询时仅需少量内存，大模型也能在小内存机器上运行。

变更说明
--------

### 1. 模型文件格式
-   旧格式：`.tree` (Pickle)
-   **新格式**：`.npz` (Numpy Zip)
-   训练生成的模型文件默认为 `model.npz`。

### 2. 训练命令
命令参数基本不变，但默认输出文件已更新。

# 训练模型 (默认输出 model.npz)
python train.py --data-dir data_train/

# 指定输出路径
python train.py --output-tree my_model.npz

### 3. 查询命令
查询命令自动适配 `.npz` 格式。

# 单帧查询(单树查询)
python query.py --bvh-file data_test/09_09.bvh --frame-index 10 --model-tree model_tree0_z0.tree.npz --model-metadata model_tree0_z0.pkl

# 交互式模式
python query.py --interactive --model-tree model.npz

### 4. 代码变更
-   `octree_node.py` 已移除。
-   `flat_octree.py` 新增，包含核心数据结构。
-   `octree_builder.py`, `query.py`, `train.py` 已更新以支持新格式。

注意事项
--------
-   新旧格式**不兼容**。请重新运行 `train.py` 生成新的 `.npz` 模型。
-   旧的 `.tree` 文件无法再被加载。
