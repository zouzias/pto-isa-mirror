#!/bin/bash

# 脚本功能：将pypto仓库中的头文件同步到pto-isa仓库的include/pypto目录

# 定义路径
PTO_ISA_DIR="/root/.openclaw/workspace/lzd-ai/pto-isa/include/pypto"
PYPTO_DIR="/root/.openclaw/workspace/lzd-ai/pypto/framework/src/interface"

# 检查目录是否存在
if [ ! -d "$PTO_ISA_DIR" ]; then
    echo "错误：pto-isa目录不存在: $PTO_ISA_DIR"
    exit 1
fi

if [ ! -d "$PYPTO_DIR" ]; then
    echo "错误：pypto目录不存在: $PYPTO_DIR"
    exit 1
fi

echo "开始同步pypto头文件到pto-isa..."

# 同步函数（从pypto目录结构开始遍历）
sync_from_pypto() {
    local pypto_dir="$1"
    local relative_path="$2"
    
    # 构建目标目录路径
    local pto_isa_dir="$PTO_ISA_DIR$relative_path"
    
    # 检查目标目录是否存在
    if [ ! -d "$pto_isa_dir" ]; then
        return
    fi
    
    echo "同步目录: $relative_path"
    
    # 拷贝当前目录下的头文件
    for pypto_file in "$pypto_dir"/*.h; do
        if [ -f "$pypto_file" ]; then
            local target_file="$pto_isa_dir/$(basename "$pypto_file")"
            echo "  拷贝: $(basename "$pypto_file")"
            cp -f "$pypto_file" "$target_file"
        fi
    done
    
    # 递归处理子目录
    for subdir in "$pypto_dir"/*; do
        if [ -d "$subdir" ]; then
            local subdir_name=$(basename "$subdir")
            local new_relative_path="$relative_path/$subdir_name"
            sync_from_pypto "$subdir" "$new_relative_path"
        fi
    done
}

# 开始同步
sync_from_pypto "$PYPTO_DIR" ""

echo "同步完成！"
