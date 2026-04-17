-- PTO Regression Database Schema

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_hash TEXT NOT NULL,
    branch TEXT,
    arch TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP,
    status TEXT DEFAULT "pending"
);

CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    test_name TEXT NOT NULL,
    arch TEXT NOT NULL,
    dtype TEXT,
    shape TEXT,
    elements INTEGER,
    total_tick INTEGER,
    vf_cycles REAL,
    pop_retire_cycles REAL,
    epc REAL,
    instr_epc REAL,
    vf_epc REAL,
    status TEXT DEFAULT "pending",
    error_msg TEXT,
    log_path TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS baselines (
    test_name TEXT NOT NULL,
    arch TEXT NOT NULL,
    baseline_epc REAL,
    baseline_commit TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (test_name, arch)
);
