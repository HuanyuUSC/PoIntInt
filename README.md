# PoIntInt: PoINT-based INTersection Volume as Boundary INTegral

本项目使用 `uv` 作为包管理器，已配置好 PyTorch CUDA 12.6 和 Kaolin 的索引源。

## 环境安装

1. 确保已安装 [uv](https://docs.astral.sh/uv/)
2. 在项目根目录执行：
   ```bash
   uv sync
   ```
## 运行脚本

示例脚本位于 `simplicits_demo.py`，用于 CLI 环境重现 Demo：

- `uv run python simplicits_demo.py --mesh assets/fox.obj --export fox_deformed.obj`
- 需要可用的 GPU（CUDA 12.6 对应驱动）以及与 `pyproject.toml` 匹配的 PyTorch、Kaolin 版本。
- 可以通过参数调整采样数、训练步数、模拟步数等；运行 `--help` 查看完整选项。

