-- Portable project-execution ledger schema.
-- This is an evidence/state database, not the runtime WAF traffic database.
CREATE TABLE IF NOT EXISTS project_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS phases (
  phase INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  score REAL,
  cutoff REAL,
  notes TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  phase INTEGER NOT NULL,
  task TEXT NOT NULL,
  status TEXT NOT NULL,
  evidence TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS test_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  phase INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  score REAL,
  details TEXT NOT NULL,
  run_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS change_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  phase INTEGER NOT NULL,
  action TEXT NOT NULL,
  target TEXT NOT NULL,
  details TEXT NOT NULL,
  created_at TEXT NOT NULL
);
