find tests/costmodel/st/testcase -name "*.cpp" | xargs clang-format -i
clang-format -i include/pto/costmodel/a2a3/*.hpp
clang-format -i include/pto/costmodel/common/*.hpp
clang-format -i include/pto/costmodel/*.hpp
clang-format -i tests/costmodel/st/common/*.hpp

