// ── Merged record: interleaved instructions + sync events ──

struct MergedEntry {
    bool is_sync = false;
    bool sync_consumed = false; // set by InlineSyncIntoMerged for consumed SIGNALs
    InstrRecord instr;
    SyncRecord sync;
    uint64_t seq = 0;
};

inline std::vector<MergedEntry> MergeRecords()
{
    auto &instrs = PtoRecorder::Get();
    auto &syncs = SyncRecorder::Get();

    std::vector<MergedEntry> merged;
    merged.reserve(instrs.size() + syncs.size());

    size_t i = 0, j = 0;
    while (i < instrs.size() && j < syncs.size()) {
        if (instrs[i].seq <= syncs[j].seq) {
            merged.push_back({false, false, instrs[i], {}, instrs[i].seq});
            i++;
        } else {
            merged.push_back({true, false, {}, syncs[j], syncs[j].seq});
            j++;
        }
    }
    while (i < instrs.size()) {
        merged.push_back({false, false, instrs[i], {}, instrs[i].seq});
        i++;
    }
    while (j < syncs.size()) {
        merged.push_back({true, false, {}, syncs[j], syncs[j].seq});
        j++;
    }
    return merged;
}

// Merge records from all logical cores belonging to one physical core
inline std::vector<MergedEntry> MergeRecordsForPhysicalCore(uint32_t phys_core)
{
    std::vector<InstrRecord> all_instrs;
    std::vector<SyncRecord> all_syncs;

    for (uint32_t sub = 0; sub < VEC_CORES_PER_AIC; ++sub) {
        uint32_t logical = phys_core * VEC_CORES_PER_AIC + sub;
        auto &instrs = PtoRecorder::GetForCore(logical);
        auto &syncs = SyncRecorder::GetForCore(logical);
        all_instrs.insert(all_instrs.end(), instrs.begin(), instrs.end());
        all_syncs.insert(all_syncs.end(), syncs.begin(), syncs.end());
    }

    std::vector<MergedEntry> merged;
    merged.reserve(all_instrs.size() + all_syncs.size());

    size_t i = 0, j = 0;
    while (i < all_instrs.size() && j < all_syncs.size()) {
        if (all_instrs[i].seq <= all_syncs[j].seq) {
            merged.push_back({false, false, all_instrs[i], {}, all_instrs[i].seq});
            i++;
        } else {
            merged.push_back({true, false, {}, all_syncs[j], all_syncs[j].seq});
            j++;
        }
    }
    while (i < all_instrs.size()) {
        merged.push_back({false, false, all_instrs[i], {}, all_instrs[i].seq});
        i++;
    }
    while (j < all_syncs.size()) {
        merged.push_back({true, false, {}, all_syncs[j], all_syncs[j].seq});
        j++;
    }
    return merged;
}

// ── Build PipeQueues from merged records ──

// Route VecCore1 AIV stages to extended queue indices 8-10
inline int QueueIndex(PipeStage stage, uint32_t subblock_id)
{
    if (subblock_id == 0 || !IsAIVStage(stage)) {
        return static_cast<int>(stage);
    }
    switch (stage) {
        case PipeStage::MTE2_AIV:
            return 8;
        case PipeStage::MTE3:
            return 9;
        case PipeStage::Vector:
            return 10;
        default:
            return static_cast<int>(stage);
    }
}

inline std::vector<PipeQueue> MakePipeQueues()
{
    std::vector<PipeQueue> qs;
    // Standard 8 queues (indices 0-7, matching PipeStage enum)
    for (int i = 0; i < static_cast<int>(PipeStage::COUNT); ++i)
        qs.emplace_back(PipeQueue{PipeStageName(static_cast<PipeStage>(i))});
    // VecCore1 extended queues (indices 8-10)
    qs.emplace_back(PipeQueue{"MTE2_AIV_1"});
    qs.emplace_back(PipeQueue{"MTE3_1"});
    qs.emplace_back(PipeQueue{"VEC_1"});
    return qs;
}

// ── Map hardware pipe_t → PipeStage (for sync event routing) ──
// PIPE_S=0, PIPE_V=1, PIPE_MTE1=2, PIPE_MTE2=3, PIPE_MTE3=4, PIPE_M=5, PIPE_ALL=6, PIPE_FIX=7

inline PipeStage SyncPipeToStage(int pipe, PipeStage last_mte2)
{
    switch (pipe) {
        case 1:
            return PipeStage::Vector;
        case 2:
            return PipeStage::MTE1;
        case 3:
            return last_mte2; // PIPE_MTE2 → context-dependent
        case 4:
            return PipeStage::MTE3;
        case 5:
            return PipeStage::Matrix;
        case 7:
            return PipeStage::Fixpipe;
        default:
            return PipeStage::Scalar;
    }
}

inline bool IsMTE2(PipeStage s)
{
    return s == PipeStage::MTE2_AIV || s == PipeStage::MTE2_AIC;
}

// ── Inline sync into merged entries (by global seq order) ──
// Merges SIGNAL/WAIT sync entries into neighboring instruction entries at the
// merged-entry level, BEFORE queue assignment. This fixes the case where the
// src_pipe of a sync doesn't match the PipeStage of the preceding instruction
// (e.g., TEXTRACT is PipeStage::Vector but sync uses PIPE_MTE1).

inline void InlineSyncIntoMerged(std::vector<MergedEntry> &merged)
{
    // Pass 1: attach SIGNALs to preceding instruction entries.
    // Track per (subblock_id, PipeStage) to separate VecCore0/VecCore1.
    constexpr int STAGE_COUNT = static_cast<int>(PipeStage::COUNT);
    int last_instr[2][STAGE_COUNT];
    std::fill_n(&last_instr[0][0], 2 * STAGE_COUNT, -1);
    PipeStage last_mte2 = PipeStage::MTE2_AIV;

    for (size_t i = 0; i < merged.size(); ++i) {
        auto &entry = merged[i];
        if (!entry.is_sync) {
            int sub = static_cast<int>(entry.instr.subblock_id);
            int pipe_idx = static_cast<int>(entry.instr.stage);
            if (sub < 2 && pipe_idx >= 0 && pipe_idx < STAGE_COUNT)
                last_instr[sub][pipe_idx] = static_cast<int>(i);
            if (IsMTE2(entry.instr.stage))
                last_mte2 = entry.instr.stage;
        } else if (entry.sync.kind == SyncKind::Signal && !entry.sync.cross_core) {
            int sub = static_cast<int>(entry.sync.subblock_id);
            event_t ch = EncodeSyncKey(entry.sync.dst_pipe, entry.sync.event_id, entry.sync.subblock_id);
            PipeStage src_stage = SyncPipeToStage(entry.sync.src_pipe, last_mte2);
            int idx = (sub < 2) ? last_instr[sub][static_cast<int>(src_stage)] : -1;
            if (idx >= 0) {
                auto &instr = merged[idx].instr;
                if (instr.extra_sync_signal_count < InstrRecord::MAX_SYNC_SIGNALS) {
                    instr.extra_sync_signals[instr.extra_sync_signal_count++] = ch;
                    entry.sync_consumed = true;
                }
            }
        }
    }

    // Pass 2: WAITs are NOT merged into instructions. They remain as standalone
    // entries so FillQueuesFromMerged places them on the correct dst_pipe queue.

    // NOTE: No structural auto-dep rules. Cross-pipe dependencies (MTE1→Matrix,
    // Matrix→Fixpipe) are handled by explicit sync (set_flag/wait_flag) in the
    // kernel code. Structural rules caused deadlock when combined with sync WAITs
    // in FIFO queues: the auto-dep wait on the Matrix queue front entry blocked
    // all subsequent entries including SIGNALs needed to resolve other pipes.
}

// ── Build PipeQueues from merged records

inline bool AppendWait(PipeEntry &entry, event_t event)
{
    if (event < 0 || entry.wait_count >= PipeEntry::MAX_WAIT)
        return false;
    entry.wait_events[entry.wait_count++] = event;
    return true;
}

inline bool AppendUniqueWait(PipeEntry &entry, event_t event)
{
    if (event < 0)
        return false;
    for (int i = 0; i < entry.wait_count; ++i) {
        if (entry.wait_events[i] == event)
            return true;
    }
    return AppendWait(entry, event);
}

inline bool AppendExtraSignal(PipeEntry &entry, event_t event)
{
    if (event < 0 || entry.extra_signal_count >= PipeEntry::MAX_SIGNALS)
        return false;
    entry.extra_signals[entry.extra_signal_count++] = event;
    return true;
}

inline bool UsesDirection(uint64_t cv_key, TPipeDir direction)
{
    return (CvKeyDirection(cv_key) & static_cast<uint8_t>(direction)) == static_cast<uint8_t>(direction);
}

inline std::string InstrName(const InstrRecord &instr, uint64_t seq, bool include_seq)
{
    std::string name = instr.opcode + "(" + std::to_string(instr.rows) + "x" + std::to_string(instr.cols) +
                       (instr.dtype.empty() ? "" : ",") + instr.dtype + ")";
    if (include_seq)
        name += ":" + std::to_string(seq);
    if (!instr.tile_args.empty())
        name += " [" + instr.tile_args + "]";
    return name;
}

inline bool IsTLoadOnStage(const InstrRecord &instr, PipeStage stage)
{
    return instr.stage == stage && instr.opcode.find("TLOAD") != std::string::npos;
}

struct CvInstrMeta {
    std::vector<event_t> push_producer_signals;
    std::vector<event_t> pop_producer_signals;
};

struct CvDependencyState {
    event_t latest_fixp_signal = -1;
    event_t latest_mte3_signal = -1;
    event_t latest_v2c_pop_signal = -1;
    std::array<uint64_t, VEC_CORES_PER_AIC> aiv_mte2_load_counts = {0, 0};
    std::vector<event_t> c2v_pop_by_aiv_load;
    std::vector<event_t> pending_c2v_producer_signals;
    std::unordered_map<uint64_t, std::deque<event_t>> ready_tokens;
    std::unordered_map<event_t, std::vector<event_t>> push_producer_signals;
    std::unordered_map<event_t, event_t> pop_by_producer_signal;
    std::unordered_set<event_t> c2v_pop_signals;

    void ObserveProducer(const InstrRecord &instr)
    {
        if (instr.stage == PipeStage::Fixpipe && instr.signal_event >= 0) {
            latest_fixp_signal = instr.signal_event;
            pending_c2v_producer_signals.push_back(instr.signal_event);
        }
        if (instr.stage == PipeStage::MTE3 && instr.signal_event >= 0) {
            latest_mte3_signal = instr.signal_event;
        }
    }

    CvInstrMeta AttachCvWaits(const InstrRecord &instr, PipeEntry &entry)
    {
        CvInstrMeta meta;
        if (instr.cv_key != 0) {
            if (instr.cv_kind == CvSyncKind::Pop)
                AttachPopWaits(instr, entry, meta);
            if (instr.cv_kind == CvSyncKind::Push)
                AttachPushWaits(instr, entry, meta);
        }
        AttachTLoadWaits(instr, entry);
        return meta;
    }

    void CommitCvSignal(const InstrRecord &instr, event_t signal_event, const CvInstrMeta &meta)
    {
        if (instr.cv_key == 0 || signal_event < 0)
            return;
        if (instr.cv_kind == CvSyncKind::Push) {
            ready_tokens[instr.cv_key].push_back(signal_event);
            if (!meta.push_producer_signals.empty())
                push_producer_signals[signal_event] = meta.push_producer_signals;
            return;
        }
        if (instr.cv_kind != CvSyncKind::Pop)
            return;
        if (UsesDirection(instr.cv_key, TPipeDir::DIR_C2V)) {
            c2v_pop_signals.insert(signal_event);
            for (auto producer_signal : meta.pop_producer_signals) {
                if (producer_signal >= 0)
                    pop_by_producer_signal[producer_signal] = signal_event;
            }
        }
        if (UsesDirection(instr.cv_key, TPipeDir::DIR_V2C)) {
            latest_v2c_pop_signal = signal_event;
        }
    }

private:
    void AttachPopWaits(const InstrRecord &instr, PipeEntry &entry, CvInstrMeta &meta)
    {
        auto &tokens = ready_tokens[instr.cv_key];
        if (!tokens.empty() && AppendWait(entry, tokens.front())) {
            event_t token = tokens.front();
            tokens.pop_front();
            auto producers = push_producer_signals.find(token);
            if (producers != push_producer_signals.end())
                meta.pop_producer_signals = producers->second;
            return;
        }
        if (UsesDirection(instr.cv_key, TPipeDir::DIR_C2V) && AppendWait(entry, latest_fixp_signal))
            meta.pop_producer_signals.push_back(latest_fixp_signal);
        if (UsesDirection(instr.cv_key, TPipeDir::DIR_V2C))
            AppendWait(entry, latest_mte3_signal);
    }

    void AttachPushWaits(const InstrRecord &instr, PipeEntry &entry, CvInstrMeta &meta)
    {
        if (UsesDirection(instr.cv_key, TPipeDir::DIR_C2V) && AppendWait(entry, latest_fixp_signal)) {
            meta.push_producer_signals = pending_c2v_producer_signals;
            if (meta.push_producer_signals.empty())
                meta.push_producer_signals.push_back(latest_fixp_signal);
            pending_c2v_producer_signals.clear();
        }
        if (UsesDirection(instr.cv_key, TPipeDir::DIR_V2C) && AppendWait(entry, latest_mte3_signal))
            meta.push_producer_signals.push_back(latest_mte3_signal);
    }

    void AttachTLoadWaits(const InstrRecord &instr, PipeEntry &entry)
    {
        if (IsTLoadOnStage(instr, PipeStage::MTE2_AIV)) {
            uint64_t load_index = 0;
            bool has_load_index = false;
            if (instr.subblock_id < aiv_mte2_load_counts.size()) {
                load_index = aiv_mte2_load_counts[instr.subblock_id]++;
                has_load_index = true;
                if (instr.subblock_id > 0 && load_index < c2v_pop_by_aiv_load.size())
                    AppendUniqueWait(entry, c2v_pop_by_aiv_load[load_index]);
            }
            for (int i = 0; i < entry.wait_count; ++i) {
                auto pop = pop_by_producer_signal.find(entry.wait_events[i]);
                if (pop != pop_by_producer_signal.end()) {
                    AppendUniqueWait(entry, pop->second);
                    break;
                }
            }
            if (has_load_index && instr.subblock_id == 0) {
                event_t c2v_pop = FindC2VPopWait(entry);
                if (c2v_pop >= 0) {
                    if (load_index >= c2v_pop_by_aiv_load.size())
                        c2v_pop_by_aiv_load.resize(load_index + 1, -1);
                    c2v_pop_by_aiv_load[load_index] = c2v_pop;
                }
            }
        }
        if (IsTLoadOnStage(instr, PipeStage::MTE2_AIC)) {
            AppendUniqueWait(entry, latest_v2c_pop_signal);
        } else if (IsTLoadOnStage(instr, PipeStage::MTE2_AIV) && entry.wait_count == 0) {
            AppendUniqueWait(entry, latest_fixp_signal);
        }
    }

    event_t FindC2VPopWait(const PipeEntry &entry) const
    {
        for (int i = 0; i < entry.wait_count; ++i) {
            if (c2v_pop_signals.count(entry.wait_events[i]) != 0)
                return entry.wait_events[i];
        }
        return -1;
    }
};

inline void AppendInstrExtraSignals(const InstrRecord &instr, PipeEntry &entry)
{
    for (int i = 0; i < instr.extra_sync_signal_count; ++i)
        AppendExtraSignal(entry, instr.extra_sync_signals[i]);
    if (instr.subblock_id != 0 || !IsAICStage(instr.stage))
        return;
    int sync_base = SyncChannelBase();
    for (int i = 0; i < instr.extra_sync_signal_count; ++i) {
        event_t sig = instr.extra_sync_signals[i];
        if (sig < sync_base)
            continue;
        int hw_dst = (sig - sync_base) / EVENTS_PER_PIPE;
        if (!IsCubeSidePipe(hw_dst))
            AppendExtraSignal(entry, sig + HW_PIPE_COUNT * EVENTS_PER_PIPE);
    }
}

inline PipeEntry BuildInstrEntry(const InstrRecord &instr, uint64_t seq, bool include_seq, CvDependencyState &cv_state)
{
    PipeEntry entry;
    entry.name = InstrName(instr, seq, include_seq);
    entry.duration = instr.estimated_cycles;
    entry.signal_event = instr.signal_event;
    entry.wait_count = instr.wait_count;
    for (int i = 0; i < entry.wait_count; ++i)
        entry.wait_events[i] = instr.wait_events[i];
    auto cv_meta = cv_state.AttachCvWaits(instr, entry);
    AppendInstrExtraSignals(instr, entry);
    entry.is_mte = IsGMAccessPipe(instr.stage);
    entry.state = (entry.wait_count > 0) ? PipeEntry::WAITING : PipeEntry::IDLE;
    cv_state.CommitCvSignal(instr, entry.signal_event, cv_meta);
    return entry;
}

inline bool IsSyncEntryName(const std::string &name)
{
    return name.find("SIGNAL(") == 0 || name.find("WAIT(") == 0 || name == "BARRIER";
}

inline PipeEvent BuildPipeEvent(const PipeEntry &entry, const std::string &pipe_label, bool stuck = false)
{
    PipeEvent event;
    event.name = entry.name;
    event.pipe_label = pipe_label;
    event.start_cycle = entry.start_cycle;
    event.end_cycle = entry.end_cycle;
    event.signal_event = entry.signal_event;
    event.wait_count = entry.wait_count;
    for (int i = 0; i < entry.wait_count; ++i)
        event.wait_events[i] = entry.wait_events[i];
    event.extra_signal_count = entry.extra_signal_count;
    for (int i = 0; i < entry.extra_signal_count; ++i)
        event.extra_signals[i] = entry.extra_signals[i];
    event.is_sync = IsSyncEntryName(entry.name);
    event.stuck = stuck;
    event.duration = entry.duration;
    return event;
}

inline void SignalChannel(std::vector<EventChannel> &channels, event_t event, uint64_t now)
{
    if (event < 0 || static_cast<size_t>(event) >= channels.size())
        return;
    auto &channel = channels[event];
    channel.count++;
    channel.signal_cycle = now;
}

inline bool AnyRunning(const std::vector<PipeQueue> &queues)
{
    for (auto &q : queues) {
        if (q.current && q.current->state == PipeEntry::RUNNING)
            return true;
    }
    return false;
}

struct StepProgress {
    bool any_active = false;
    bool dispatched = false;
    uint64_t next_event = UINT64_MAX;
};

inline void CompleteRunningEntry(PipeQueue &queue, PipeTimeline &timeline, std::vector<EventChannel> &channels,
                                 uint64_t now, StepProgress &progress)
{
    if (!queue.current || queue.current->state != PipeEntry::RUNNING)
        return;
    progress.any_active = true;
    if (now < queue.current->end_cycle) {
        progress.next_event = std::min(progress.next_event, queue.current->end_cycle);
        return;
    }
    timeline.events.push_back(BuildPipeEvent(*queue.current, queue.label));
    SignalChannel(channels, queue.current->signal_event, now);
    for (int si = 0; si < queue.current->extra_signal_count; ++si)
        SignalChannel(channels, queue.current->extra_signals[si], now);
    queue.current->state = PipeEntry::DONE;
    queue.current = nullptr;
    queue.next_free_cycle = now;
}

inline void PopDoneEntries(std::vector<PipeQueue> &queues)
{
    for (auto &queue : queues) {
        while (!queue.entries.empty() && queue.entries.front().state == PipeEntry::DONE)
            queue.entries.pop_front();
    }
}

inline bool SingleCoreWaitsReady(const PipeEntry &entry, const std::vector<EventChannel> &channels)
{
    for (int i = 0; i < entry.wait_count; ++i) {
        event_t wait = entry.wait_events[i];
        if (wait >= 0 && (static_cast<size_t>(wait) >= channels.size() || !channels[wait].signaled()))
            return false;
    }
    return true;
}

inline void ConsumeSingleCoreSyncWaits(const PipeEntry &entry, std::vector<EventChannel> &channels)
{
    int sync_base = SyncChannelBase();
    for (int i = 0; i < entry.wait_count; ++i) {
        event_t wait = entry.wait_events[i];
        if (wait >= sync_base && static_cast<size_t>(wait) < channels.size())
            channels[wait].count--;
    }
}

inline void StartEntry(PipeQueue &queue, PipeEntry &entry, uint64_t now, L2CacheModel &cache, StepProgress &progress)
{
    uint64_t duration = entry.duration;
    if (entry.is_mte && entry.size > 0) {
        auto result_cache = cache.Access(entry.addr, entry.size, duration);
        duration = result_cache.adjusted_duration;
    }
    entry.state = PipeEntry::RUNNING;
    entry.start_cycle = std::max(queue.next_free_cycle, now);
    entry.end_cycle = entry.start_cycle + duration;
    queue.current = &entry;
    progress.any_active = true;
    progress.dispatched = true;
}

inline void TryDispatchSingleCore(PipeQueue &queue, std::vector<EventChannel> &channels, L2CacheModel &cache,
                                  uint64_t now, StepProgress &progress)
{
    if (!queue.CanAccept() || queue.entries.empty())
        return;
    auto &entry = queue.entries.front();
    if (entry.name == "BARRIER") {
        StartEntry(queue, entry, now, cache, progress);
        return;
    }
    if (entry.wait_count > 0 && entry.state == PipeEntry::WAITING) {
        if (!SingleCoreWaitsReady(entry, channels)) {
            progress.any_active = true;
            return;
        }
        ConsumeSingleCoreSyncWaits(entry, channels);
        entry.state = PipeEntry::IDLE;
    }
    if (entry.state == PipeEntry::IDLE)
        StartEntry(queue, entry, now, cache, progress);
}

inline void CollectStuckEntries(const std::vector<PipeQueue> &queues, PipeTimeline &timeline)
{
    for (auto &queue : queues) {
        for (auto &entry : queue.entries)
            timeline.events.push_back(BuildPipeEvent(entry, queue.label, true));
    }
}

constexpr int EVENTS_PER_CORE = 8192;        // max channels per core (auto-dep + sync)
constexpr int CROSS_CHANNEL_OFFSET = 262144; // must be > max_cores * EVENTS_PER_CORE

inline event_t EncodeMergedSyncEvent(const SyncRecord &sync, bool cross_core_enabled)
{
    if (cross_core_enabled && sync.cross_core) {
        return CROSS_CHANNEL_OFFSET + sync.event_id;
    }
    return EncodeSyncKey(sync.dst_pipe, sync.event_id, sync.subblock_id);
}

inline void PushBarrierEntry(std::vector<PipeQueue> &queues)
{
    PipeEntry entry;
    entry.name = "BARRIER";
    entry.duration = 0;
    queues[static_cast<int>(PipeStage::Scalar)].Push(std::move(entry));
}

inline void PushSignalEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage last_mte2,
                            bool include_seq, bool cross_core_enabled)
{
    const auto &sync = entry.sync;
    PipeEntry pipe_entry;
    pipe_entry.name = "SIGNAL(" + std::to_string(sync.event_id) + ")";
    if (include_seq) {
        pipe_entry.name += ":" + std::to_string(entry.seq);
    }
    pipe_entry.duration = 0;
    pipe_entry.signal_event = EncodeMergedSyncEvent(sync, cross_core_enabled);
    if (!sync.cross_core && sync.subblock_id == 0 && IsCubeSidePipe(sync.src_pipe) && !IsCubeSidePipe(sync.dst_pipe)) {
        AppendExtraSignal(pipe_entry, EncodeSyncKey(sync.dst_pipe, sync.event_id, 1));
    }
    queues[QueueIndex(SyncPipeToStage(sync.src_pipe, last_mte2), sync.subblock_id)].Push(std::move(pipe_entry));
}

inline void PushWaitEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage last_mte2,
                          bool include_seq, bool cross_core_enabled)
{
    const auto &sync = entry.sync;
    PipeEntry pipe_entry;
    pipe_entry.name = "WAIT(" + std::to_string(sync.event_id) + ")";
    if (include_seq) {
        pipe_entry.name += ":" + std::to_string(entry.seq);
    }
    pipe_entry.duration = 0;
    pipe_entry.is_sync_wait = true;
    AppendWait(pipe_entry, EncodeMergedSyncEvent(sync, cross_core_enabled));
    pipe_entry.state = PipeEntry::WAITING;
    queues[QueueIndex(SyncPipeToStage(sync.dst_pipe, last_mte2), sync.subblock_id)].Push(std::move(pipe_entry));
}

inline void PushInstrEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage &last_mte2,
                           bool include_seq, CvDependencyState &cv_state)
{
    int idx = QueueIndex(entry.instr.stage, entry.instr.subblock_id);
    if (idx < 0 || idx >= static_cast<int>(queues.size())) {
        return;
    }
    if (IsMTE2(entry.instr.stage)) {
        last_mte2 = entry.instr.stage;
    }
    cv_state.ObserveProducer(entry.instr);
    queues[idx].Push(BuildInstrEntry(entry.instr, entry.seq, include_seq, cv_state));
}

inline void PushMergedEntry(std::vector<PipeQueue> &queues, const MergedEntry &entry, PipeStage &last_mte2,
                            bool include_seq, bool cross_core_enabled, CvDependencyState &cv_state)
{
    if (!entry.is_sync) {
        PushInstrEntry(queues, entry, last_mte2, include_seq, cv_state);
        return;
    }
    switch (entry.sync.kind) {
        case SyncKind::Barrier:
            PushBarrierEntry(queues);
            break;
        case SyncKind::Signal:
            PushSignalEntry(queues, entry, last_mte2, include_seq, cross_core_enabled);
            break;
        case SyncKind::Wait:
            PushWaitEntry(queues, entry, last_mte2, include_seq, cross_core_enabled);
            break;
    }
}

inline void FillQueuesFromMergedImpl(std::vector<PipeQueue> &queues, const std::vector<MergedEntry> &merged,
                                     bool include_seq, bool cross_core_enabled)
{
    PipeStage last_mte2 = PipeStage::MTE2_AIV;
    CvDependencyState cv_state;
    for (auto &entry : merged) {
        if (entry.sync_consumed) {
            continue;
        }
        PushMergedEntry(queues, entry, last_mte2, include_seq, cross_core_enabled, cv_state);
    }
}

inline void FillQueuesFromMerged(std::vector<PipeQueue> &queues, std::vector<EventChannel> &channels,
                                 const std::vector<MergedEntry> &merged)
{
    (void)channels;
    FillQueuesFromMergedImpl(queues, merged, true, false);
}

// ── Event-driven Step simulation with cache ──

inline PipeTimeline StepSimulate(std::vector<PipeQueue> &queues, std::vector<EventChannel> &channels,
                                 L2CacheModel &cache, uint32_t max_steps = 500000)
{
    PipeTimeline result;
    uint64_t now = 0;

    for (uint32_t step = 0; step < max_steps; ++step) {
        StepProgress progress;
        for (auto &queue : queues)
            CompleteRunningEntry(queue, result, channels, now, progress);

        PopDoneEntries(queues);

        for (auto &queue : queues)
            TryDispatchSingleCore(queue, channels, cache, now, progress);

        if (!progress.any_active)
            break;
        if (!progress.dispatched) {
            if (!AnyRunning(queues))
                break;
            if (progress.next_event != UINT64_MAX && progress.next_event > now)
                now = progress.next_event;
        }
    }

    CollectStuckEntries(queues, result);

    for (auto &ev : result.events)
        result.total_cycles = std::max(result.total_cycles, ev.end_cycle);
    return result;
}

// ── RunPipeline: main simulation entry point ──

inline PipeTimeline RunPipeline()
{
    auto merged = MergeRecords();
    InlineSyncIntoMerged(merged);
    auto queues = MakePipeQueues();
    std::vector<EventChannel> channels(TotalChannels());
    FillQueuesFromMerged(queues, channels, merged);

    L2CacheModel cache;
    return StepSimulate(queues, channels, cache);
}

// ── Multi-core simulation data structures ──

// Helper: lookup EventChannel for multi-core simulation
inline EventChannel *LookupChannel(std::vector<EventChannel> &intra, std::vector<EventChannel> &cross, uint32_t core_id,
                                   event_t event_id)
{
    if (event_id < 0)
        return nullptr;
    if (event_id >= CROSS_CHANNEL_OFFSET) {
        int idx = event_id - CROSS_CHANNEL_OFFSET;
        if (idx < static_cast<int>(cross.size()))
            return &cross[idx];
        return nullptr;
    }
    int idx = core_id * EVENTS_PER_CORE + event_id;
    if (idx < static_cast<int>(intra.size()))
        return &intra[idx];
    return nullptr;
}

inline void SignalChannel(EventChannel *channel, uint64_t now)
{
    if (!channel)
        return;
    channel->count++;
    channel->signal_cycle = now;
}

struct CorePipeline {
    uint32_t core_id;
    std::vector<PipeQueue> queues; // 7 pipes per core
};

struct MultiCoreTimeline {
    std::vector<PipeTimeline> per_core;
    uint64_t total_cycles = 0;
};

inline bool AnyRunning(const std::vector<CorePipeline> &core_pipelines)
{
    for (auto &core : core_pipelines) {
        if (AnyRunning(core.queues))
            return true;
    }
    return false;
}

inline void CompleteRunningEntry(CorePipeline &core, PipeQueue &queue, PipeTimeline &timeline,
                                 std::vector<EventChannel> &intra_channels, std::vector<EventChannel> &cross_channels,
                                 uint64_t now, StepProgress &progress)
{
    if (!queue.current || queue.current->state != PipeEntry::RUNNING)
        return;
    progress.any_active = true;
    if (now < queue.current->end_cycle) {
        progress.next_event = std::min(progress.next_event, queue.current->end_cycle);
        return;
    }
    timeline.events.push_back(BuildPipeEvent(*queue.current, queue.label));
    SignalChannel(LookupChannel(intra_channels, cross_channels, core.core_id, queue.current->signal_event), now);
    for (int si = 0; si < queue.current->extra_signal_count; ++si)
        SignalChannel(LookupChannel(intra_channels, cross_channels, core.core_id, queue.current->extra_signals[si]),
                      now);
    queue.current->state = PipeEntry::DONE;
    queue.current = nullptr;
    queue.next_free_cycle = now;
}

inline bool MultiCoreWaitsReady(const PipeEntry &entry, uint32_t core_id, std::vector<EventChannel> &intra_channels,
                                std::vector<EventChannel> &cross_channels)
{
    for (int i = 0; i < entry.wait_count; ++i) {
        auto *channel = LookupChannel(intra_channels, cross_channels, core_id, entry.wait_events[i]);
        if (channel && !channel->signaled())
            return false;
    }
    return true;
}

inline void ConsumeMultiCoreSyncWaits(const PipeEntry &entry, uint32_t core_id,
                                      std::vector<EventChannel> &intra_channels,
                                      std::vector<EventChannel> &cross_channels)
{
    int sync_base = SyncChannelBase();
    for (int i = 0; i < entry.wait_count; ++i) {
        if (entry.wait_events[i] < sync_base)
            continue;
        auto *channel = LookupChannel(intra_channels, cross_channels, core_id, entry.wait_events[i]);
        if (channel)
            channel->count--;
    }
}

inline void TryDispatchMultiCore(CorePipeline &core, PipeQueue &queue, std::vector<EventChannel> &intra_channels,
                                 std::vector<EventChannel> &cross_channels, L2CacheModel &cache, uint64_t now,
                                 StepProgress &progress)
{
    if (!queue.CanAccept() || queue.entries.empty())
        return;
    auto &entry = queue.entries.front();
    if (entry.name == "BARRIER") {
        StartEntry(queue, entry, now, cache, progress);
        return;
    }
    if (entry.wait_count > 0 && entry.state == PipeEntry::WAITING) {
        if (!MultiCoreWaitsReady(entry, core.core_id, intra_channels, cross_channels)) {
            progress.any_active = true;
            return;
        }
        ConsumeMultiCoreSyncWaits(entry, core.core_id, intra_channels, cross_channels);
        entry.state = PipeEntry::IDLE;
    }
    if (entry.state == PipeEntry::IDLE)
        StartEntry(queue, entry, now, cache, progress);
}

inline void CollectStuckEntries(const std::vector<CorePipeline> &core_pipelines, MultiCoreTimeline &timeline)
{
    for (size_t c = 0; c < core_pipelines.size(); ++c) {
        for (auto &queue : core_pipelines[c].queues) {
            for (auto &entry : queue.entries)
                timeline.per_core[c].events.push_back(BuildPipeEvent(entry, queue.label, true));
        }
    }
}

// ── Per-physical-core merge (combines subblock logical cores) ──

inline std::vector<std::vector<MergedEntry>> MergeRecordsPerCore(uint32_t num_phys_cores)
{
    std::vector<std::vector<MergedEntry>> result(num_phys_cores);
    for (uint32_t c = 0; c < num_phys_cores; ++c) {
        result[c] = MergeRecordsForPhysicalCore(c);
    }
    return result;
}

// ── Multi-core FillQueuesFromMerged (handles cross_core offset) ──

inline void FillQueuesFromMerged(std::vector<PipeQueue> &queues, std::vector<EventChannel> & /*intra_channels*/,
                                 std::vector<EventChannel> & /*cross_channels*/, const std::vector<MergedEntry> &merged)
{
    FillQueuesFromMergedImpl(queues, merged, false, true);
}

// ── Event-driven multi-core step simulation ──

inline MultiCoreTimeline StepSimulateMultiCore(std::vector<CorePipeline> &core_pipelines,
                                               std::vector<EventChannel> &intra_channels,
                                               std::vector<EventChannel> &cross_channels, L2CacheModel &cache,
                                               uint32_t max_steps = 500000)
{
    MultiCoreTimeline result;
    result.per_core.resize(core_pipelines.size());
    uint64_t now = 0;

    for (uint32_t step = 0; step < max_steps; ++step) {
        StepProgress progress;
        for (auto &cp : core_pipelines) {
            for (auto &queue : cp.queues)
                CompleteRunningEntry(cp, queue, result.per_core[cp.core_id], intra_channels, cross_channels, now,
                                     progress);
        }

        for (auto &cp : core_pipelines)
            PopDoneEntries(cp.queues);

        for (auto &cp : core_pipelines) {
            for (auto &queue : cp.queues)
                TryDispatchMultiCore(cp, queue, intra_channels, cross_channels, cache, now, progress);
        }

        if (!progress.any_active)
            break;
        if (!progress.dispatched) {
            if (!AnyRunning(core_pipelines))
                break; // true deadlock
        }
        if (progress.next_event != UINT64_MAX && progress.next_event > now)
            now = progress.next_event;
    }

    CollectStuckEntries(core_pipelines, result);

    for (auto &tl : result.per_core) {
        for (auto &ev : tl.events)
            tl.total_cycles = std::max(tl.total_cycles, ev.end_cycle);
        result.total_cycles = std::max(result.total_cycles, tl.total_cycles);
    }
    return result;
}
