+ PTO（Parallel Tile Operation）是昇腾 CANN 定义的一套面向 tile 编程的虚拟 ISA。本仓库提供 PTO Tile 指令的实现、示例、测试与文档，帮助开发者在不同昇腾代际之间更平滑地迁移和优化算子。
+ 其中面向不同的硬件架构，对指令的实现方式可能不同
+ 目前的硬件架构及对应的指令实现以及测试用例目录如下：

|硬件架构|指令|测试用例|
|---|---|---|
|A2A3|include/pto/npu/a2a3|tests/npu/a2a3|
|A5|include/pto/npu/a5|tests/npu/a5|
|kirin9030|include/pto/npu/kirin9030|tests/npu/kirin9030|
|kirinX90|include/pto/npu/kirinX90|tests/npu/kirin9030|

+ 由于A5、kirin9030、kirinX90硬件架构相似，因此kirin9030和kirinX90可能会复用A5指令的实现，kirinX90也可能复用kirin9030指令的实现，复用的指令在include/pto/npu/kirin9030/header.hpp和include/pto/npu/kirinX90/header.hpp文件中记录
+ kirinX90和kirin9030共用一套测试用例，但二者目前已经支持的指令有明显差异

+ 请按这种方式区分指令：如存在include/pto/npu/a5/TDeQuant.hpp文件，那么TDeQuant就作为一个被统计的指令，当然，如果在header.cpp中复用了指令，也要被统计在内

+ 任务的目标是将kirinX90支持的指令和kirin9030对齐

