# TPUSH/TPOP 增强设计（摘要）

本文档为《Enhanced TPUSH/TPOP ISA Design for Intra-Cluster Function Group Data Communication》的中文摘要版。

## 目标

- 支持簇内函数组之间的高效数据传递
- 在不破坏现有语义的前提下增强 TPUSH/TPOP 协议
- 提供可扩展的标记（tag）与双通道 FIFO 机制

## 设计要点

- 基于环形 FIFO 的生产者/消费者协议
- 通过 tag 机制区分数据流与控制流
- 支持更细粒度的同步与可观测性

## 适用范围

- 面向 NPU 端 TPipe/TMPipe 相关场景
- 主要用于跨流水或跨执行单元的数据接力

## 说明

当前页为中文摘要入口，详细规范与流程图请参考英文原文：
[HL_ptoisa_newfeature20260306_TPUSH_TPOP.md](./HL_ptoisa_newfeature20260306_TPUSH_TPOP.md)。
