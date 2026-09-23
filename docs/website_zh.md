# 文档网站（MkDocs）

本仓库可以使用 **MkDocs** 和 **Read the Docs** 主题作为静态文档站点浏览。

MkDocs 配置位于 `docs/mkdocs/` 目录下，设置为浏览整个仓库的 markdown 文件（包括 `kernels/`、`tests/` 等目录下的 README 文件）。

## 前置要求

- Python >= 3.9.x

> **备注：** Python 宣布 3.7.x/3.8.x 已经 EOL，CANN 即将停止对该版本的支持，请升级到 Python >= 3.9.x 的版本。

## 安装（推荐：使用虚拟环境）

```bash
python3 -m venv .venv-mkdocs
source .venv-mkdocs/bin/activate
python -m pip install --upgrade pip
python -m pip install -r docs/mkdocs/requirements.txt
```

## 本地服务（实时重载）

```bash
python -m mkdocs serve -f docs/mkdocs/mkdocs.yml
```

然后打开 `http://127.0.0.1:8000/`。

## 构建静态站点

```bash
python -m mkdocs build -f docs/mkdocs/mkdocs.yml
```

输出将写入 `site/` 目录（如果传递 `-d` 参数，则写入自定义目录）。

## 使用 CMake 构建

你可以将静态站点构建作为 CMake 构建的一部分：

```bash
cmake -S docs -B build/docs
cmake --build build/docs --target pto_docs
```

站点将生成在 `build/docs/site/` 目录下。

本地服务：

```bash
cmake --build build/docs --target pto_docs_serve
```

## 注意事项

- MkDocs 源目录是 `docs/mkdocs/src/`。
- `docs/mkdocs/gen_pages.py` 在构建时将仓库 markdown 文件镜像到站点中，保留路径以便仓库相对链接继续工作。

## 内存布局详图

TLOAD 步长回退、TINSERT NZ split 和 TEXTRACT NZ 配图由
`docs/tools/gen_memory_layout_svgs.py` 生成，中英文文档共用 SVG。
修改生成脚本后，重新生成配图并检查产物是否一致：

```bash
python3 docs/tools/gen_memory_layout_svgs.py
python3 docs/tools/gen_memory_layout_svgs.py --check
```

配图输出到 `docs/figures/isa/`，可通过 `--output-dir` 指定其他目录。
