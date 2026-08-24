/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

constexpr size_t kBufferSize = 64 * 64 * sizeof(float);

bool WriteZeroFile(const std::filesystem::path& path)
{
    const std::array<char, kBufferSize> zeros{};
    std::ofstream output(path, std::ios::binary);
    output.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    return output.good();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 2;
    }
    const std::filesystem::path caseDir(argv[1]);
    std::filesystem::create_directories(caseDir);
    if (!WriteZeroFile(caseDir / "input1.bin") || !WriteZeroFile(caseDir / "input2.bin") ||
        !WriteZeroFile(caseDir / "golden.bin")) {
        return 1;
    }
    return 0;
}
