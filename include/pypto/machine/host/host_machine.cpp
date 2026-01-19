/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file host_machine.cpp
 * \brief
 */

#include "host_machine.h"
#include <functional>
#include <unistd.h>
#include <dlfcn.h>
#include "tilefwk/tilefwk.h"
#include "interface/inner/tilefwk.h"
#include "interface/function/function.h"
#include "interface/program/program.h"
#include "interface/utils/op_info_manager.h"

extern "C" {
using RunPassFunc = int (*)(npu::tile_fwk::Program &, npu::tile_fwk::Function &, const std::string &);
using GetResumePathFunc = std::string (*)(const std::string &);
using ExecuteFunc = int (*)(npu::tile_fwk::MachineTask *, npu::tile_fwk::FunctionCache &);
using PlatformFunc = std::string (*)();
using MatchCacheFunc = bool (*)(const std::string &);
using InitFunc = int (*)();

struct Backend {
    RunPassFunc runPass;
    GetResumePathFunc getResumePath;
    ExecuteFunc execute;
    ExecuteFunc simuExecute;
    PlatformFunc platform;
    MatchCacheFunc matchCache;

    static Backend &GetBackend() {
        static Backend backend;
        return backend;
    }

    ~Backend() {
        if (compilerHandle != nullptr) {
            dlclose(compilerHandle);
        }
        if (simuHandle != nullptr) {
            dlclose(simuHandle);
        }
    }

private:
    Backend() {
        progHandle = dlopen(nullptr, RTLD_LAZY | RTLD_NOLOAD);
        compilerHandle = dlopen("libtile_fwk_compiler.so", RTLD_LAZY | RTLD_NOLOAD);
        simuHandle = dlopen("libtile_fwk_simulator.so", RTLD_LAZY | RTLD_NOLOAD);

        runPass = (RunPassFunc)GetSymbol(progHandle, "RunPass");
        getResumePath = (GetResumePathFunc)GetSymbol(progHandle, "GetResumePath");
        execute = (ExecuteFunc)GetSymbol(compilerHandle, "Execute");
        platform = (PlatformFunc)GetSymbol(compilerHandle, "GetPlatformInfo");
        matchCache = (MatchCacheFunc)GetSymbol(compilerHandle, "MatchCache");
        simuExecute = (ExecuteFunc)GetSymbol(simuHandle, "ExecuteSimulation");

        auto initFunc = (InitFunc)GetSymbol(compilerHandle, "Initialize");
        if (initFunc) {
            initFunc();
        }
    }

    void *GetSymbol(void *handle, const char *sym) {
        void *ptr = nullptr;
        if (handle != nullptr) {
            ptr = dlsym(handle, sym);
        }
        if (ptr == nullptr) {
            ptr = dlsym(progHandle, sym);
        }
        return ptr;
    }

    void *progHandle;
    void *compilerHandle;
    void *simuHandle;
};
}

namespace npu::tile_fwk {
namespace {
enum class StashType {
    Function = 0,
    ProgramConfig,
    InternalConfig,
    ConfigJson
};
}

HostMachine& HostMachine::GetInstance() {
    static HostMachine sHostMachine;
    return sHostMachine;
}

/* 支持模式转换 */
bool HostMachine::Init(const HostMachineMode mode) {
    if (initialized_.load() && mode == mode_) {
        ALOG_DEBUG("HostMachine is already initialized.");
        return true;
    }
    if (mode_ == HostMachineMode::SERVER && mode == HostMachineMode::API) {
        DestroyThread();
    }
    mode_ = mode;

    if (mode == HostMachineMode::SERVER) {
        InitThread();
    }

    initialized_.store(true);
    return true;
}

void HostMachine::Destroy() {
    if (mode_ == HostMachineMode::SERVER) {
        WaitTaskFinish();
        DestroyThread();
    }

    ALOG_DEBUG("HostMachine is destroying...");
}

void HostMachine::InitThread() {
    stopFlag_.store(false);
    for (int idx = 0; idx < compileThreadCount_; ++idx) {
        compileThreads_.emplace_back(&HostMachine::CompileThreadFunc, this);
    }
    for (int idx = 0; idx < agentThreadCount_; ++idx) {
        agentThreads_.emplace_back(&HostMachine::AgentThreadFunc, this);
    }
}

void HostMachine::DestroyThread() {
    auto notifyThread = [](std::mutex &mutex, std::condition_variable &cv) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.notify_all();
    };
    stopFlag_.store(true);
    notifyThread(compileQueueMutex_, compileQueueCv_);
    for (auto &thread : compileThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    notifyThread(agentQueueMutex_, agentQueueCv_);
    for (auto &thread : agentThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    compileThreads_.clear();
    agentThreads_.clear();
}

void HostMachine::CompileFunction(Function* func) const {
    auto &backend = Backend::GetBackend();
    if (!func->HasCallOperation() && backend.runPass) {
        ALOG_INFO_F("RunPass function %s", func->GetMagicName().c_str());
        ASSERT(backend.runPass(Program::GetInstance(), *func, config::GetPassStrategy())) << "Run pass failed.";
    }
    if (func->IsFunctionType(FunctionType::DYNAMIC) || func->IsFunctionTypeAndGraphType(FunctionType::STATIC, GraphType::TILE_GRAPH)) {
        auto path = config::GetAbsoluteTopFolder() + "/program.json";
        Program::GetInstance().DumpJsonFile(path);
        config::SetRunDataOption(KEY_PROGRAM_PATH, path);
    }
    if (func->rootFunc_ != nullptr) {
        func->rootFunc_->DumpTopoFile(config::LogTopFolder() + "/topo.json");
    }
}

void HostMachine::SubTask(Function *function) {
    if (mode_ == HostMachineMode::API) {
        if (curTask != nullptr) {
            ALOG_WARN("CurTask is already running.");
        }
        MACHINE_ASSERT(curTask == nullptr);
        curTask = new MachineTask(curTaskId_++, function);
        return;
    }

    std::lock_guard<std::mutex> lock(compileQueueMutex_);
    auto task = std::make_unique<MachineTask>(curTaskId_++, function);
    compileQueue_.Push(std::move(task));
    compileQueueCv_.notify_one(); // 通知编译线程
}

void HostMachine::WaitTaskFinish() {
    while (curTaskId_ != finishQueue_.Size()) {
        usleep(1000); // sleep 1000 us
    } // wait all task finish
    ALOG_DEBUG_F("Finish all host machine task count: %lu.", curTaskId_.load());

    /* reset counter */
    curTaskId_ = 0;
    while (!finishQueue_.Empty()) {
        auto task = finishQueue_.Pop();
        auto &error = task->Error();
        if (!error.empty()) {
            finishQueue_.Clear();
            throw std::runtime_error(error);
        }
    }
}

void HostMachine::StashTask(Function* function) {
    if (function == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(stashQueueMutex_);
    stashedFuncQueue_.Push(std::make_tuple(function,
        config::Duplicate(),
        ConfigManager::Instance().GetInternalConfig(),
        ConfigManager::Instance().GetJsonData()));
}

void HostMachine::SubAllStashedTask() {
    std::lock_guard<std::mutex> lock(stashQueueMutex_);
    while (!stashedFuncQueue_.Empty()) {
        auto funcData = stashedFuncQueue_.Pop();
        config::Restore(std::get<static_cast<size_t>(StashType::ProgramConfig)>(funcData));
        ConfigManager::Instance().SetInternalConfig(std::get<static_cast<size_t>(StashType::InternalConfig)>(funcData));
        ConfigManager::Instance().SetJsonData(std::get<static_cast<size_t>(StashType::ConfigJson)>(funcData));
        SubTask(std::get<0>(funcData));
        WaitTaskFinish();
    }
}

void HostMachine::ClearStashFuncQueue() {
    stashedFuncQueue_.Clear();
}

std::string HostMachine::GetCacheKeyFromFunction(Function *function) {
    std::string cacheKey;
    if (function == nullptr) {
        return cacheKey;
    }
    if (function->BelongTo().GetLastFunction() != nullptr &&
        function->BelongTo().GetLastFunction()->GetFunctionType() == FunctionType::DYNAMIC) {
        cacheKey = function->BelongTo().GetLastFunction()->GetFunctionHash().Data();
        OpInfoManager::GetInstance().GetOpFuncName() = function->BelongTo().GetLastFunction()->GetMagicName() +
                                                        cacheKey;
    } else {
        cacheKey = function->GetFunctionHash().Data();
    }
    return cacheKey;
}

MachineTask *HostMachine::Compile(MachineTask *task) const {
    MachineTask *compileTask = task;
    if (compileTask == nullptr) {
        if (curTask == nullptr) {   
            ALOG_WARN("Compile task is null.");
        }
        MACHINE_ASSERT(curTask != nullptr);
        compileTask = curTask;
    }
    std::string jsonPath;
    auto &backend = Backend::GetBackend();
    if (backend.getResumePath) {
        jsonPath = backend.getResumePath(config::GetPassStrategy());
    }
    bool existResumeFile = !jsonPath.empty() && (access(jsonPath.c_str(), F_OK) == 0);
    if (existResumeFile) {
        std::ifstream file(jsonPath);
        ASSERT(file.good()) << "Json file: " << jsonPath << " open failed!!!";
        Json jsonData;
        try {
            file >> jsonData;
        } catch (const std::exception &e) {
            ASSERT(false) << "Json file: " << jsonPath << " parsing error: " << e.what();
        }
        Program::GetInstance().LoadJson(jsonData);
        Function *func = Program::GetInstance().GetCurrentFunction();

        CompileFunction(func);
        compileTask->SetFunction(func);
    } else {
        auto function = compileTask->GetFunction();
        compileTask->SetCacheKey(GetCacheKeyFromFunction(function));
        if (backend.matchCache && backend.matchCache(compileTask->GetCacheKey())) {
            compileTask->SetCacheReuseType(CacheReuseType::Bin);
        } else {
            CompileFunction(function);
        }
    }

    return compileTask;
}

void HostMachine::PushAgentQueue(std::unique_ptr<MachineTask> task) {
    std::lock_guard<std::mutex> lock(agentQueueMutex_);
    agentQueue_.Push(std::move(task));
    agentQueueCv_.notify_one(); // 通知代理线程
}

void HostMachine::CompileThreadFunc() {
    while (!stopFlag_.load()) {
        std::unique_ptr<MachineTask> task;
        std::unique_lock<std::mutex> lock(compileQueueMutex_);
        compileQueueCv_.wait(lock, [this] { return !compileQueue_.Empty() || stopFlag_.load(); });
        if (stopFlag_.load()) {
            break;
        }
        task = compileQueue_.Pop();
        lock.unlock();

        try {
            (void)Compile(task.get());
        } catch (const Error &e) {
            task->SetError(e.what());
            PushFinishQueue(std::move(task));
            return;
        }

        PushAgentQueue(std::move(task));
    }
}

void HostMachine::PushFinishQueue(std::unique_ptr<MachineTask> task) {
    finishQueue_.Push(std::move(task));
}

void HostMachine::AgentThreadFunc() {
    while (!stopFlag_.load()) {
        std::unique_ptr<MachineTask> task;
        std::unique_lock<std::mutex> lock(agentQueueMutex_);
        agentQueueCv_.wait(lock, [this] { return !agentQueue_.Empty() || stopFlag_.load(); });
        if (stopFlag_.load()) {
            break;
        }
        task = agentQueue_.Pop();
        lock.unlock();

        try {
            auto &cache = Program::GetInstance().GetFunctionCache();
            auto &backend = Backend::GetBackend();
            if (backend.simuExecute && config::GetPlatformConfig(KEY_ENABLE_COST_MODEL, true)) {
                ALOG_INFO_F("Simulate function %s", task->GetFunction()->GetMagicName().c_str());
                backend.simuExecute(task.get(), cache);
            }
            if (backend.execute && config::GetPlatformConfig(KEY_ENABLE_AIHAC_BACKEND, true)) {
                ALOG_INFO_F("Compile function %s", task->GetFunction()->GetMagicName().c_str());
                backend.execute(task.get(), cache);
            }
        } catch (const std::exception &e) {
            task->SetError(std::move(e.what()));
        }
        PushFinishQueue(std::move(task));
    }
}

std::string HostMachine::GetPlatformInfo() const {
    auto &backend = Backend::GetBackend();
    if (backend.platform == nullptr) {
        ALOG_ERROR("Backend platform symbol GetPlatformInfo not found.");
        return "";
    }
    return backend.platform();
}
} // namespace npu::tile_fwk
