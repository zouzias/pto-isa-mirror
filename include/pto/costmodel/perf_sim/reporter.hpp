/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_REPORTER_HPP
#define PTO_PERF_SIM_REPORTER_HPP

#include "pipe_model.hpp"
#include "tile_dep_tracker.hpp"
#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

namespace pto::perf_sim {

// ── Simulation report ──

struct SimReport {
    std::string op_name;
    uint32_t num_cores = 1;
    PipeTimeline timeline;            // single-core path
    MultiCoreTimeline multi_timeline; // multi-core path
    uint64_t instr_count = 0;
    uint64_t sync_count = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    double cache_hit_rate = 0.0;
};

// ── PerfSimReporter: run simulation, print text, write JSON ──

class PerfSimReporter {
public:
    // Fixed track layout for Chrome Trace (1AIC + 2AIV per core)
    struct TrackInfo {
        const char *tid;          // queue label (must match PipeQueue label)
        std::string display_name; // [AIC-{core_id}] PIPE or [AIV-{sub}] PIPE
        const char *group;        // AIC / AIV-0 / AIV-1
    };

    static std::vector<TrackInfo> GetTracksForCore(uint32_t core_id)
    {
        std::string aic = "[AIC-" + std::to_string(core_id) + "] ";
        std::string aiv0 = "[AIC-" + std::to_string(core_id) + "/AIV-0] ";
        std::string aiv1 = "[AIC-" + std::to_string(core_id) + "/AIV-1] ";
        return {
            {"Scalar", aic + "Scalar", "AIC"},      {"MTE2_AIC", aic + "MTE2", "AIC"},
            {"MTE1", aic + "MTE1", "AIC"},          {"CUBE", aic + "CUBE", "AIC"},
            {"FIXP", aic + "FIXP", "AIC"},          {"MTE2_AIV", aiv0 + "MTE2", "AIV-0"},
            {"VEC", aiv0 + "VEC", "AIV-0"},         {"MTE3", aiv0 + "MTE3", "AIV-0"},
            {"MTE2_AIV_1", aiv1 + "MTE2", "AIV-1"}, {"VEC_1", aiv1 + "VEC", "AIV-1"},
            {"MTE3_1", aiv1 + "MTE3", "AIV-1"},
        };
    }

    struct PipelineSummaryRow {
        uint32_t core_id = 0;
        std::string unit;
        uint64_t total_cycles = 0;
        uint64_t active_start_cycle = 0;
        uint64_t active_end_cycle = 0;
        uint64_t active_cycles = 0;
        uint64_t busy_cycles = 0;
        uint64_t scalar_cycles = 0;
        uint64_t mte2_aic_cycles = 0;
        uint64_t mte2_aiv_cycles = 0;
        uint64_t mte1_cycles = 0;
        uint64_t cube_cycles = 0;
        uint64_t fixp_cycles = 0;
        uint64_t vec_cycles = 0;
        uint64_t mte3_cycles = 0;
    };

    static uint64_t EventBusyCycles(const PipeEvent &event)
    {
        return event.stuck ? event.duration : (event.end_cycle - event.start_cycle);
    }

    static PipelineSummaryRow BuildPipelineSummaryRow(const PipeTimeline &timeline, uint32_t core_id,
                                                      const std::string &unit, const std::vector<std::string> &pipes)
    {
        std::unordered_map<std::string, uint64_t> busy_cycles;
        uint64_t active_start = std::numeric_limits<uint64_t>::max();
        uint64_t active_end = 0;
        for (auto &ev : timeline.events) {
            if (std::find(pipes.begin(), pipes.end(), ev.pipe_label) == pipes.end())
                continue;
            uint64_t busy = EventBusyCycles(ev);
            busy_cycles[ev.pipe_label] += busy;
            if (busy > 0) {
                active_start = std::min(active_start, ev.start_cycle);
                active_end = std::max(active_end, ev.end_cycle);
            }
        }

        auto get = [&](const char *pipe) -> uint64_t {
            auto it = busy_cycles.find(pipe);
            return it == busy_cycles.end() ? 0 : it->second;
        };

        PipelineSummaryRow row;
        row.core_id = core_id;
        row.unit = unit;
        row.total_cycles = timeline.total_cycles;
        row.active_start_cycle = active_start == std::numeric_limits<uint64_t>::max() ? 0 : active_start;
        row.active_end_cycle = active_end;
        row.active_cycles = active_end > row.active_start_cycle ? active_end - row.active_start_cycle : 0;
        row.scalar_cycles = get("Scalar");
        row.mte2_aic_cycles = get("MTE2_AIC");
        row.mte2_aiv_cycles = get("MTE2_AIV") + get("MTE2_AIV_1");
        row.mte1_cycles = get("MTE1");
        row.cube_cycles = get("CUBE");
        row.fixp_cycles = get("FIXP");
        row.vec_cycles = get("VEC") + get("VEC_1");
        row.mte3_cycles = get("MTE3") + get("MTE3_1");
        row.busy_cycles = row.scalar_cycles + row.mte2_aic_cycles + row.mte2_aiv_cycles + row.mte1_cycles +
                          row.cube_cycles + row.fixp_cycles + row.vec_cycles + row.mte3_cycles;
        return row;
    }

    static std::vector<PipelineSummaryRow> BuildPipelineSummary(const SimReport &report)
    {
        std::vector<PipelineSummaryRow> rows;
        rows.reserve(report.num_cores * 3);
        auto append_core = [&](const PipeTimeline &timeline, uint32_t core_id) {
            rows.push_back(
                BuildPipelineSummaryRow(timeline, core_id, "AIC", {"Scalar", "MTE2_AIC", "MTE1", "CUBE", "FIXP"}));
            rows.push_back(BuildPipelineSummaryRow(timeline, core_id, "AIV0", {"MTE2_AIV", "VEC", "MTE3"}));
            rows.push_back(BuildPipelineSummaryRow(timeline, core_id, "AIV1", {"MTE2_AIV_1", "VEC_1", "MTE3_1"}));
        };
        if (report.num_cores == 1)
            append_core(report.timeline, 0);
        else
            for (uint32_t c = 0; c < report.num_cores; ++c)
                append_core(report.multi_timeline.per_core[c], c);
        return rows;
    }

    SimReport Run(const std::string &op_name)
    {
        SimReport report;
        report.op_name = op_name;
        auto &cfg = GetConfig();

        if (cfg.block_dim <= 1) {
            // Single-core path: merge both logical cores (Cube+VecCore0, VecCore1)
            report.num_cores = 1;
            for (uint32_t sub = 0; sub < VEC_CORES_PER_AIC; ++sub) {
                report.instr_count += PtoRecorder::GetForCore(sub).size();
                report.sync_count += SyncRecorder::GetForCore(sub).size();
            }

            auto merged = MergeRecordsForPhysicalCore(0);
            // Single-core: cross-core sync events (ffts_cross_core_sync,
            // wait_flag_dev) are no-ops.  Filter them out to prevent
            // EncodeSyncKey(dst_pipe=-1) WAITs from deadlocking the Scalar queue.
            merged.erase(std::remove_if(merged.begin(), merged.end(),
                                        [](const MergedEntry &e) { return e.is_sync && e.sync.cross_core; }),
                         merged.end());
            InlineSyncIntoMerged(merged);
            auto queues = MakePipeQueues();
            std::vector<EventChannel> channels(TotalChannels());
            FillQueuesFromMerged(queues, channels, merged);

            L2CacheModel cache;
            report.timeline = StepSimulate(queues, channels, cache);

            report.cache_hits = cache.GetStats().hits;
            report.cache_misses = cache.GetStats().misses;
            report.cache_hit_rate = cache.GetStats().HitRate();
        } else {
            // Multi-core path
            uint32_t num_cores = cfg.block_dim;
            report.num_cores = num_cores;
            uint32_t total_logical = num_cores * VEC_CORES_PER_AIC;

            for (uint32_t lc = 0; lc < total_logical; ++lc) {
                report.instr_count += PtoRecorder::GetForCore(lc).size();
                report.sync_count += SyncRecorder::GetForCore(lc).size();
            }

            auto per_core_merged = MergeRecordsPerCore(num_cores);

            std::vector<CorePipeline> core_pipelines(num_cores);
            for (uint32_t c = 0; c < num_cores; ++c) {
                core_pipelines[c].core_id = c;
                core_pipelines[c].queues = MakePipeQueues();
            }

            std::vector<EventChannel> intra_channels(num_cores * EVENTS_PER_CORE);
            std::vector<EventChannel> cross_channels(cfg.cross_core_channel_count);

            for (uint32_t c = 0; c < num_cores; ++c) {
                // Filter cross-core sync events (CV buffer operations,
                // not real inter-core sync — same as single-core path).
                per_core_merged[c].erase(
                    std::remove_if(per_core_merged[c].begin(), per_core_merged[c].end(),
                                   [](const MergedEntry &e) { return e.is_sync && e.sync.cross_core; }),
                    per_core_merged[c].end());
                InlineSyncIntoMerged(per_core_merged[c]);
                FillQueuesFromMerged(core_pipelines[c].queues, intra_channels, cross_channels, per_core_merged[c]);
            }

            L2CacheModel cache; // shared across all cores
            report.multi_timeline = StepSimulateMultiCore(core_pipelines, intra_channels, cross_channels, cache);

            report.cache_hits = cache.GetStats().hits;
            report.cache_misses = cache.GetStats().misses;
            report.cache_hit_rate = cache.GetStats().HitRate();
        }

        return report;
    }

    // ── Pipeline table: rows=core, cols=pipe, cell=busy cycles ──
    static void PrintPipelineTable(const SimReport &report, uint64_t total_cycles, std::ostream &os)
    {
        // Fixed column order for all pipelines across all cores
        const char *const pipe_names[] = {"Scalar", "MTE2(AIC)", "MTE1", "CUBE", "FIXP", "MTE2(AIV)", "VEC", "MTE3"};
        constexpr int kPipes = 8;

        auto rows = BuildPipelineSummary(report);

        // Header row
        os << "Pipeline     |";
        for (uint32_t c = 0; c < report.num_cores; ++c)
            os << " AIC-" << c << "   |";
        os << "\n";
        os << "-------------+";
        for (uint32_t c = 0; c < report.num_cores; ++c)
            os << "--------+";
        os << "\n";

        // Data rows
        for (int p = 0; p < kPipes; ++p) {
            os << std::setw(12) << pipe_names[p] << " |";
            for (uint32_t c = 0; c < report.num_cores; ++c) {
                const auto &aic = rows[c * 3];
                const auto &aiv0 = rows[c * 3 + 1];
                const auto &aiv1 = rows[c * 3 + 2];
                const uint64_t values[kPipes] = {
                    aic.scalar_cycles,
                    aic.mte2_aic_cycles,
                    aic.mte1_cycles,
                    aic.cube_cycles,
                    aic.fixp_cycles,
                    aiv0.mte2_aiv_cycles + aiv1.mte2_aiv_cycles,
                    aiv0.vec_cycles + aiv1.vec_cycles,
                    aiv0.mte3_cycles + aiv1.mte3_cycles,
                };
                uint64_t v = values[p];
                if (v == 0) {
                    os << "       - |";
                } else {
                    os << std::setw(7) << v << " |";
                }
            }
            os << "\n";
        }
        os << "\n";
        os << "Note: AIV pipeline rows sum AIV0 and AIV1. Use pipeline_summary.csv for per-AIV active/busy cycles.\n\n";
    }

    static void PrintText(const SimReport &report, std::ostream &os = std::cout)
    {
        os << "===== Perf-Sim Report: " << report.op_name << " =====\n";
        os << "Cores        : " << report.num_cores << "\n";
        os << "Instructions : " << report.instr_count << "\n";
        os << "Sync events  : " << report.sync_count << "\n";

        uint64_t total_cycles =
            (report.num_cores == 1) ? report.timeline.total_cycles : report.multi_timeline.total_cycles;
        os << "Total cycles : " << total_cycles << "\n";
        os << "L2 Cache hits: " << report.cache_hits << "  misses: " << report.cache_misses
           << "  hit rate: " << report.cache_hit_rate << "%\n\n";

        PrintPipelineTable(report, total_cycles, os);

        os << "===== End Report =====\n";
    }

    static void WriteSwimlaneJson(const std::string &path, const SimReport &report)
    {
        // Ensure output directory exists
        auto last_slash = path.rfind('/');
        if (last_slash != std::string::npos) {
            std::string dir = path.substr(0, last_slash);
#ifdef _WIN32
            _mkdir(dir.c_str());
#else
            system(("mkdir -p " + dir).c_str());
#endif
        }

        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "[perf_sim] Cannot write JSON to: " << path << "\n";
            return;
        }

        // Chrome Trace Event format (Swimlane view with flow arrows)
        // Two passes: 1) collect signal map, 2) write X-events + flow events

        // Build per-core label map: short pipe_label -> display name with prefix
        auto build_label_map = [](int pid) {
            std::unordered_map<std::string, std::string> m;
            for (auto &t : GetTracksForCore(static_cast<uint32_t>(pid)))
                m[t.tid] = t.display_name;
            return m;
        };
        std::unordered_map<int, std::unordered_map<std::string, std::string>> label_maps;
        if (report.num_cores == 1) {
            label_maps[0] = build_label_map(0);
        } else {
            for (uint32_t c = 0; c < report.num_cores; ++c)
                label_maps[static_cast<int>(c)] = build_label_map(static_cast<int>(c));
        }

        struct FlowSrc {
            int pid;
            std::string tid;
            std::string name;
            uint64_t ts;
        };
        std::unordered_map<int64_t, std::vector<FlowSrc>> signal_map;
        auto flow_key = [](int pid, event_t ev) -> int64_t { return static_cast<int64_t>(pid) * 1000000LL + ev; };

        auto display_tid = [&](int pid, const std::string &label) -> std::string {
            auto it = label_maps.find(pid);
            if (it != label_maps.end()) {
                auto mit = it->second.find(label);
                if (mit != it->second.end())
                    return mit->second;
            }
            return label;
        };

        auto collect_signals = [&](int pid, const std::vector<PipeEvent> &events) {
            auto add_signal = [&](event_t signal, const PipeEvent &ev) {
                if (signal < 0)
                    return;
                signal_map[flow_key(pid, signal)].push_back(
                    {pid, display_tid(pid, ev.pipe_label), ev.name, ev.end_cycle});
            };
            for (auto &ev : events) {
                add_signal(ev.signal_event, ev);
                for (int i = 0; i < ev.extra_signal_count; ++i)
                    add_signal(ev.extra_signals[i], ev);
            }
        };
        if (report.num_cores == 1) {
            collect_signals(0, report.timeline.events);
        } else {
            for (uint32_t c = 0; c < report.num_cores; ++c)
                collect_signals(static_cast<int>(c), report.multi_timeline.per_core[c].events);
        }

        // Build sort_index map for ordering events by track
        auto build_sort_map = [](int pid) {
            std::unordered_map<std::string, int> m;
            auto tracks = GetTracksForCore(static_cast<uint32_t>(pid));
            for (int i = 0; i < static_cast<int>(tracks.size()); ++i)
                m[tracks[i].tid] = i;
            return m;
        };

        auto find_flow_source = [&](int pid, event_t wait, uint64_t wait_start) -> const FlowSrc * {
            auto it = signal_map.find(flow_key(pid, wait));
            if (it == signal_map.end())
                return nullptr;
            const FlowSrc *best = nullptr;
            for (const auto &src : it->second) {
                if (src.ts > wait_start)
                    continue;
                if (src.name.find("TASSIGN(") == 0 || src.name.find("TRESHAPE(") == 0 ||
                    src.name.find("TALLOC(") == 0 || src.name.find("TFREE(") == 0)
                    continue;
                if (best == nullptr || src.ts > best->ts)
                    best = &src;
            }
            return best;
        };

        auto write_json_str = [&]() {
            bool first = false; // metadata events already written
            int64_t flow_id = 1;
            auto write_ev = [&](int pid, const PipeEvent &ev) {
                // Skip zero-duration sync events (SIGNAL/WAIT) to reduce clutter
                if (ev.end_cycle == ev.start_cycle && ev.is_sync)
                    return;
                // Skip zero-duration Scalar metadata (TASSIGN, TRESHAPE, TALLOC, etc.)
                if (ev.end_cycle == ev.start_cycle && ev.pipe_label == "Scalar")
                    return;

                std::string dtid = display_tid(pid, ev.pipe_label);

                if (!first)
                    out << ",\n";
                first = false;
                out << "  {\"name\": \"" << ev.name << "\","
                    << "\"cat\": \"" << dtid << "\","
                    << "\"ph\": \"X\","
                    << "\"ts\": " << ev.start_cycle << ","
                    << "\"dur\": " << (ev.stuck ? ev.duration : (ev.end_cycle - ev.start_cycle)) << ","
                    << "\"pid\": " << pid << ","
                    << "\"tid\": \"" << dtid << "\","
                    << "\"args\":{\"signal_event\":" << ev.signal_event << ",\"wait_events\":\"";
                for (int wi = 0; wi < ev.wait_count; ++wi) {
                    if (wi > 0)
                        out << ";";
                    out << ev.wait_events[wi];
                }
                out << "\"}}";

                // Flow events: draw arrows for cross-pipe wait dependencies
                for (int i = 0; i < ev.wait_count; ++i) {
                    auto *src = find_flow_source(pid, ev.wait_events[i], ev.start_cycle);
                    if (src == nullptr)
                        continue;
                    // Only draw flow if cross-pipe (src tid != dst tid)
                    if (src->tid == dtid)
                        continue;
                    out << ",\n";
                    // Flow start (arrow source) at the signal instruction's end time
                    out << "  {\"name\": \"dep\", \"cat\": \"flow\", \"ph\": \"s\","
                        << "\"id\": " << flow_id << ","
                        << "\"pid\": " << src->pid << ","
                        << "\"tid\": \"" << src->tid << "\","
                        << "\"ts\": " << src->ts << ","
                        << "\"args\":{\"src\":\"" << src->name << "\"}}";
                    out << ",\n";
                    // Flow finish (arrow target) at this instruction's start time
                    out << "  {\"name\": \"dep\", \"cat\": \"flow\", \"ph\": \"f\","
                        << "\"id\": " << flow_id << ","
                        << "\"pid\": " << pid << ","
                        << "\"tid\": \"" << dtid << "\","
                        << "\"ts\": " << ev.start_cycle << ","
                        << "\"args\":{\"dst\":\"" << ev.name << "\"}}";
                    flow_id++;
                }
            };

            // Sort events per core by (timestamp, track_sort_index) so Perfetto
            // sees tracks in AIC→AIV order from first event appearance.
            auto write_sorted = [&](int pid, const std::vector<PipeEvent> &events) {
                auto smap = build_sort_map(pid);
                std::vector<size_t> idx(events.size());
                for (size_t i = 0; i < idx.size(); ++i)
                    idx[i] = i;
                std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
                    if (events[a].start_cycle != events[b].start_cycle)
                        return events[a].start_cycle < events[b].start_cycle;
                    int sa = smap.count(events[a].pipe_label) ? smap[events[a].pipe_label] : 99;
                    int sb = smap.count(events[b].pipe_label) ? smap[events[b].pipe_label] : 99;
                    return sa < sb;
                });
                for (auto i : idx)
                    write_ev(pid, events[i]);
            };

            if (report.num_cores == 1) {
                write_sorted(0, report.timeline.events);
            } else {
                for (uint32_t c = 0; c < report.num_cores; ++c)
                    write_sorted(static_cast<int>(c), report.multi_timeline.per_core[c].events);
            }
        };
        out << "[\n";

        // Write metadata events for fixed track ordering (per core)
        auto write_meta = [&](int pid) {
            auto tracks = GetTracksForCore(static_cast<uint32_t>(pid));
            int sort_idx = 0;
            for (auto &t : tracks) {
                // thread_name metadata
                out << ",\n";
                out << "  {\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":\"" << t.display_name
                    << "\",\"name\":\"thread_name\""
                    << ",\"args\":{\"name\":\"" << t.display_name << "\"}}";
                // thread_sort_index to enforce track ordering
                out << ",\n";
                out << "  {\"ph\":\"M\",\"pid\":" << pid << ",\"tid\":\"" << t.display_name
                    << "\",\"name\":\"thread_sort_index\""
                    << ",\"args\":{\"sort_index\":" << sort_idx++ << "}}";
            }
        };
        if (report.num_cores == 1) {
            // Process name for core 0
            out << "  {\"ph\":\"M\",\"pid\":0,\"tid\":0,"
                << "\"name\":\"process_name\",\"args\":{\"name\":\"Core 0\"}}";
            out << ",\n  {\"ph\":\"M\",\"pid\":0,\"tid\":0,"
                << "\"name\":\"process_sort_index\",\"args\":{\"sort_index\":0}}";
            write_meta(0);
        } else {
            for (uint32_t c = 0; c < report.num_cores; ++c) {
                if (c > 0)
                    out << ",";
                out << "\n  {\"ph\":\"M\",\"pid\":" << c << ",\"tid\":0,\"name\":\"process_name\""
                    << ",\"args\":{\"name\":\"Core " << c << "\"}}";
                out << ",\n  {\"ph\":\"M\",\"pid\":" << c << ",\"tid\":0,\"name\":\"process_sort_index\""
                    << ",\"args\":{\"sort_index\":" << c << "}}";
                write_meta(static_cast<int>(c));
            }
        }

        write_json_str();

        out << "\n]\n";
        out.close();
    }

    static void WritePipelineSummaryCSV(const std::string &path, const SimReport &report)
    {
        auto last_slash = path.rfind('/');
        if (last_slash != std::string::npos) {
            std::string dir = path.substr(0, last_slash);
#ifdef _WIN32
            _mkdir(dir.c_str());
#else
            system(("mkdir -p " + dir).c_str());
#endif
        }

        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "[perf_sim] Cannot write CSV to: " << path << "\n";
            return;
        }

        out << "op_name,core_id,unit,total_cycles,active_start_cycle,active_end_cycle,active_cycles,busy_cycles,"
            << "scalar_cycles,mte2_aic_cycles,mte2_aiv_cycles,mte1_cycles,cube_cycles,fixp_cycles,vec_cycles,"
            << "mte3_cycles\n";

        for (const auto &row : BuildPipelineSummary(report)) {
            out << report.op_name << "," << row.core_id << "," << row.unit << "," << row.total_cycles << ","
                << row.active_start_cycle << "," << row.active_end_cycle << "," << row.active_cycles << ","
                << row.busy_cycles << "," << row.scalar_cycles << "," << row.mte2_aic_cycles << ","
                << row.mte2_aiv_cycles << "," << row.mte1_cycles << "," << row.cube_cycles << "," << row.fixp_cycles
                << "," << row.vec_cycles << "," << row.mte3_cycles << "\n";
        }
        out.close();
    }

    static void WriteDependencyDOT(const std::string &path, const SimReport &report)
    {
        auto last_slash = path.rfind('/');
        if (last_slash != std::string::npos) {
            std::string dir = path.substr(0, last_slash);
#ifndef _WIN32
            system(("mkdir -p " + dir).c_str());
#endif
        }

        std::ofstream out(path);
        if (!out.is_open()) {
            std::cerr << "[perf_sim] Cannot write DOT to: " << path << "\n";
            return;
        }

        out << "digraph pipeline {\n";
        out << "  rankdir=LR;\n";
        out << "  node [shape=box, style=filled, fontsize=8, "
            << "fontname=\"Courier\"];\n";
        out << "  edge [color=\"#888888\", arrowsize=0.6];\n\n";

        auto pipe_color = [](const std::string &pipe) -> const char * {
            if (pipe == "Scalar")
                return "\"#D5D5D5\"";
            if (pipe == "MTE2_AIV")
                return "\"#ADD8E6\"";
            if (pipe == "MTE2_AIC")
                return "\"#87CEEB\"";
            if (pipe == "MTE1")
                return "\"#00CED1\"";
            if (pipe == "MTE3")
                return "\"#FFA500\"";
            if (pipe == "VEC")
                return "\"#90EE90\"";
            if (pipe == "CUBE")
                return "\"#FFD700\"";
            if (pipe == "FIXP")
                return "\"#FFB6C1\"";
            return "white";
        };

        // Collect filtered nodes
        struct NodeInfo {
            int dot_id;
            std::string label;
            std::string color;
            std::string pipe;
            event_t wait_events[8];
            int wait_count;
            bool stuck;
            bool is_sync;
        };

        std::vector<NodeInfo> nodes;
        std::unordered_map<event_t, int> signal_to_node; // signal_event -> nodes index
        int dot_id = 0;

        auto process_events = [&](int pid, const std::vector<PipeEvent> &events) {
            for (auto &ev : events) {
                // Only skip zero-duration Scalar metadata (TASSIGN, TRESHAPE, etc.)
                if (ev.end_cycle == ev.start_cycle && ev.pipe_label == "Scalar" && !ev.is_sync)
                    continue;

                int idx = static_cast<int>(nodes.size());
                NodeInfo ni;
                ni.dot_id = dot_id++;
                std::string core_prefix = (report.num_cores > 1) ? "C" + std::to_string(pid) + " " : "";
                ni.label =
                    core_prefix + ev.name + "\\n" + std::to_string(ev.start_cycle) + "-" + std::to_string(ev.end_cycle);
                ni.color = ev.stuck ? "\"#FF4444\"" : pipe_color(ev.pipe_label);
                ni.pipe = ev.pipe_label;
                ni.wait_count = ev.wait_count;
                for (int i = 0; i < ev.wait_count; ++i)
                    ni.wait_events[i] = ev.wait_events[i];
                ni.stuck = ev.stuck;
                ni.is_sync = ev.is_sync;

                nodes.push_back(std::move(ni));
                if (ev.signal_event >= 0)
                    signal_to_node[ev.signal_event] = idx;
            }
        };

        if (report.num_cores == 1) {
            process_events(0, report.timeline.events);
        } else {
            for (uint32_t c = 0; c < report.num_cores; ++c)
                process_events(static_cast<int>(c), report.multi_timeline.per_core[c].events);
        }

        // Group nodes by pipe label for subgraph clusters
        std::unordered_map<std::string, std::vector<int>> pipe_groups;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
            pipe_groups[nodes[i].pipe].push_back(i);

        for (auto &[pipe_name, indices] : pipe_groups) {
            out << "  subgraph \"cluster_" << pipe_name << "\" {\n";
            out << "    label=\"" << pipe_name << "\";\n";
            out << "    style=filled;\n";
            out << "    color=\""
                << (pipe_name == "Scalar" ?
                        "#EEEEEE" :
                        pipe_name == "MTE2_AIV" ?
                        "#E8F4F8" :
                        pipe_name == "MTE2_AIC" ?
                        "#E0F0F8" :
                        pipe_name == "MTE1" ?
                        "#E0FFFF" :
                        pipe_name == "MTE3" ?
                        "#FFF0E0" :
                        pipe_name == "VEC" ?
                        "#E8FFE8" :
                        pipe_name == "CUBE" ? "#FFFFF0" : pipe_name == "FIXP" ? "#FFF0F5" : "#F0F0F0")
                << "\";\n";
            for (int idx : indices) {
                auto &ni = nodes[idx];
                out << "    n" << ni.dot_id << " [label=\"" << ni.label << "\", fillcolor=" << ni.color;
                if (ni.is_sync)
                    out << ", shape=ellipse, width=0.6, height=0.3";
                if (ni.stuck)
                    out << ", fontcolor=\"white\", penwidth=2";
                out << "];\n";
            }
            out << "  }\n\n";
        }

        // Write dependency edges
        for (auto &ni : nodes) {
            for (int i = 0; i < ni.wait_count; ++i) {
                auto it = signal_to_node.find(ni.wait_events[i]);
                if (it != signal_to_node.end()) {
                    out << "  n" << nodes[it->second].dot_id << " -> n" << ni.dot_id << ";\n";
                }
            }
        }

        // Legend
        out << "\n  subgraph cluster_legend {\n";
        out << "    label=\"Legend\";\n";
        out << "    style=filled;\n";
        out << "    color=\"white\";\n";
        out << "    node [shape=plaintext, style=\"\"];\n";
        out << "    leg [label=<<TABLE BORDER=\"0\" CELLBORDER=\"0\">"
            << "<TR><TD BGCOLOR=\"#90EE90\">VEC</TD>"
            << "<TD BGCOLOR=\"#FFD700\">CUBE</TD>"
            << "<TD BGCOLOR=\"#ADD8E6\">MTE2_AIV</TD>"
            << "<TD BGCOLOR=\"#87CEEB\">MTE2_AIC</TD>"
            << "<TD BGCOLOR=\"#00CED1\">MTE1</TD>"
            << "<TD BGCOLOR=\"#FFA500\">MTE3</TD>"
            << "<TD BGCOLOR=\"#FFB6C1\">FIXP</TD>"
            << "<TD BGCOLOR=\"#FF4444\"><FONT COLOR=\"white\">STUCK</FONT></TD>"
            << "</TR></TABLE>>];\n";
        out << "  }\n";

        out << "}\n";
        out.close();
    }

    // ── Sync Edge Audit: count SIGNAL/WAIT per channel, check balance ──

    static const char *HwPipeLabel(int hw_pipe)
    {
        static const char *names[] = {"S", "VEC", "MTE1", "MTE2", "MTE3", "CUBE", "ALL", "FIXP"};
        return (hw_pipe >= 0 && hw_pipe < 8) ? names[hw_pipe] : "?";
    }

    struct EdgeStat {
        std::unordered_map<std::string, int> src_counts; // src_pipe -> signal count
        int hw_dst = -1;
        int evt_id = -1;
        int sig = 0;
        int wait = 0;
    };

    static void PrintSyncEdgeAudit(const std::vector<PipeEvent> &events, std::ostream &os)
    {
        int sync_base = SyncChannelBase();
        std::unordered_map<event_t, EdgeStat> stats;

        for (auto &ev : events) {
            bool is_sig = (ev.name.find("SIGNAL(") == 0);
            bool is_wait_ev = (ev.name.find("WAIT(") == 0);

            // Standalone SIGNAL
            if (is_sig && ev.signal_event >= sync_base) {
                auto &s = stats[ev.signal_event];
                s.src_counts[ev.pipe_label]++;
                int off = ev.signal_event - sync_base;
                s.hw_dst = off / EVENTS_PER_PIPE;
                s.evt_id = off % EVENTS_PER_PIPE;
                s.sig++;
            }
            // Standalone WAIT
            if (is_wait_ev) {
                for (int i = 0; i < ev.wait_count; ++i) {
                    if (ev.wait_events[i] >= sync_base) {
                        auto &s = stats[ev.wait_events[i]];
                        int off = ev.wait_events[i] - sync_base;
                        s.hw_dst = off / EVENTS_PER_PIPE;
                        s.evt_id = off % EVENTS_PER_PIPE;
                        s.wait++;
                    }
                }
            }
            // Inlined extra_signals (SIGNALs merged into instructions)
            for (int i = 0; i < ev.extra_signal_count; ++i) {
                if (ev.extra_signals[i] >= sync_base) {
                    auto &s = stats[ev.extra_signals[i]];
                    s.src_counts[ev.pipe_label]++;
                    int off = ev.extra_signals[i] - sync_base;
                    s.hw_dst = off / EVENTS_PER_PIPE;
                    s.evt_id = off % EVENTS_PER_PIPE;
                    s.sig++;
                }
            }
            // Sync waits in regular instructions
            if (!is_sig && !is_wait_ev) {
                for (int i = 0; i < ev.wait_count; ++i) {
                    if (ev.wait_events[i] >= sync_base) {
                        auto &s = stats[ev.wait_events[i]];
                        int off = ev.wait_events[i] - sync_base;
                        s.hw_dst = off / EVENTS_PER_PIPE;
                        s.evt_id = off % EVENTS_PER_PIPE;
                        s.wait++;
                    }
                }
            }
        }

        // Sort by channel key
        std::vector<std::pair<event_t, EdgeStat>> sorted(stats.begin(), stats.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        os << "\n-- Sync Edge Audit --\n";
        os << "  #   Ch    -> Dst   Flag  SIG  WAIT  OK  Sources\n";
        int n = 0;
        int tot_sig = 0, tot_wait = 0, unbal = 0;
        for (auto &[ch, st] : sorted) {
            bool ok = (st.sig == st.wait);
            if (!ok)
                unbal++;
            tot_sig += st.sig;
            tot_wait += st.wait;

            // Build sources string: "Fixpipe:7 + MTE1:8"
            std::string src_str;
            std::vector<std::pair<std::string, int>> src_sorted(st.src_counts.begin(), st.src_counts.end());
            std::sort(src_sorted.begin(), src_sorted.end(),
                      [](const auto &a, const auto &b) { return a.second > b.second; });
            for (size_t i = 0; i < src_sorted.size(); ++i) {
                if (i > 0)
                    src_str += " + ";
                src_str += src_sorted[i].first + ":" + std::to_string(src_sorted[i].second);
            }

            char line[256];
            snprintf(line, sizeof(line), "  %2d  %3d   -> %-4s   %d    %3d  %4d   %s   %s\n", ++n, ch,
                     HwPipeLabel(st.hw_dst), st.evt_id, st.sig, st.wait, ok ? "OK" : "!!", src_str.c_str());
            os << line;
        }
        os << "  Total: " << tot_sig << " SIGNAL, " << tot_wait << " WAIT, " << unbal << " unbalanced\n";
    }

private:
    static void PrintCoreDetail(const PipeTimeline &timeline, uint64_t total_cycles, uint32_t core_id, std::ostream &os)
    {
        // Collect pipe stats by label
        std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> pipe_stats;
        for (auto &ev : timeline.events) {
            auto &[busy, count] = pipe_stats[ev.pipe_label];
            busy += ev.stuck ? ev.duration : (ev.end_cycle - ev.start_cycle);
            count++;
        }

        // Print in fixed track order (AIC → AIV-0 → AIV-1)
        auto tracks = GetTracksForCore(core_id);
        os << "Pipe utilization:\n";
        for (auto &t : tracks) {
            auto it = pipe_stats.find(t.tid);
            if (it == pipe_stats.end())
                continue;
            auto &[busy, count] = it->second;
            double util = total_cycles > 0 ? 100.0 * busy / total_cycles : 0.0;
            os << "  " << t.display_name << " : " << count << " ops, busy " << busy << " cycles (" << util << "%)\n";
        }
        // Print any pipes not in tracks (e.g., Scalar)
        for (auto &[name, stats] : pipe_stats) {
            bool found = false;
            for (auto &t : tracks)
                if (t.tid == name) {
                    found = true;
                    break;
                }
            if (found)
                continue;
            auto &[busy, count] = stats;
            double util = total_cycles > 0 ? 100.0 * busy / total_cycles : 0.0;
            os << "  " << name << " : " << count << " ops, busy " << busy << " cycles (" << util << "%)\n";
        }

        os << "\nTimeline:\n";
        for (auto &ev : timeline.events) {
            os << "  [" << ev.pipe_label << "] " << ev.name << "  " << ev.start_cycle << " - " << ev.end_cycle << " ("
               << (ev.stuck ? ev.duration : (ev.end_cycle - ev.start_cycle)) << " cycles)\n";
        }
        os << "\n";
    }
};

} // namespace pto::perf_sim

#endif
