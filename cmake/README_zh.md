# CMake/

本目录放仓库用到的 CMake 辅助模块，主要覆盖打包流程与第三方依赖集成。

## 目录内容

- 与打包相关的 CMake 逻辑（`package` 目标）
- 通过 FetchContent 复用 CANN 工程公共 CMake 仓库
- `CMake/third_party/` 下的第三方依赖下载/集成脚本

## 关键文件

- `CMake/package.CMake`：由顶层 `CMakeLists.txt` 引入的打包入口函数，调用工程仓库提供的 `set_cann_cpack_config` 完成 CPack 配置
- `CMake/fetch_cann_cmake.CMake`：拉取 `https://gitcode.com/cann/CMake`，并暴露公共函数（`init_cann_project`、`set_cann_cpack_config`、内置 `makeself` 及 install 脚本等）
- `CMake/func.CMake`：项目内本地辅助函数（protobuf、签名、打包）
- `CMake/third_party/`：第三方依赖辅助脚本

## 入口

- 顶层 `CMakeLists.txt` 会引入 `CMake/package.CMake` 并调用相关打包辅助函数
- `build.sh --pkg` 会触发仓库的打包流程
