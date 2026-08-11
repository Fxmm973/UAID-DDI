#!/usr/bin/env python
# coding=utf-8
"""
Audit logger: records invocation metadata and hashes for reproducibility.
Import this in every table-generation script to ensure consistent audit trails.
"""
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone


def get_git_commit(repo_root=None):
    """Return current git commit hash, or 'unknown' if not in a git repo."""
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def sha256_file(filepath):
    """Return SHA256 hex digest of a file, or 'NOT_FOUND' if missing."""
    if not os.path.exists(filepath):
        return 'NOT_FOUND'
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


class AuditLogger:
    """Context manager / helper for table-generation scripts."""

    def __init__(self, script_name, input_files=None, repo_root=None):
        self.script_name = script_name
        self.input_files = input_files or []
        if repo_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.repo_root = repo_root
        self.audit_dir = os.path.join(repo_root, 'audit', 'table_logs')
        self.start_time = None
        self.metadata = {}

    def start(self):
        self.start_time = datetime.now(timezone.utc)
        os.makedirs(self.audit_dir, exist_ok=True)
        self.metadata = {
            'script': self.script_name,
            'timestamp_utc': self.start_time.isoformat(),
            'git_commit': get_git_commit(self.repo_root),
            'command_line': ' '.join(sys.argv),
            'python_version': sys.version,
            'input_files': {},
        }
        for fpath in self.input_files:
            abs_path = os.path.join(self.repo_root, fpath)
            self.metadata['input_files'][fpath] = sha256_file(abs_path)
        return self

    def finish(self, output_text=None, output_files=None):
        end_time = datetime.now(timezone.utc)
        self.metadata['end_timestamp_utc'] = end_time.isoformat()
        self.metadata['duration_seconds'] = (end_time - self.start_time).total_seconds()
        self.metadata['output_files'] = {}
        if output_files:
            for fpath in output_files:
                abs_path = os.path.join(self.repo_root, fpath)
                self.metadata['output_files'][fpath] = sha256_file(abs_path)
        json_path = os.path.join(self.audit_dir, f'{self.script_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        log_path = os.path.join(self.audit_dir, f'{self.script_name}.log')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"=== {self.script_name} Audit Log ===\n")
            f.write(f"Timestamp (UTC): {self.metadata['timestamp_utc']}\n")
            f.write(f"Duration: {self.metadata['duration_seconds']:.1f}s\n")
            f.write(f"Git commit: {self.metadata['git_commit']}\n")
            f.write(f"Command: {self.metadata['command_line']}\n")
            f.write(f"Python: {self.metadata['python_version']}\n\n")
            f.write("Input file hashes:\n")
            for fname, fhash in self.metadata['input_files'].items():
                f.write(f"  {fname}: {fhash}\n")
            f.write("\nOutput file hashes:\n")
            for fname, fhash in self.metadata.get('output_files', {}).items():
                f.write(f"  {fname}: {fhash}\n")
            if output_text:
                f.write("\n--- Table Output ---\n")
                f.write(output_text)
        logging.info('Audit log written to %s', log_path)
        logging.info('Audit metadata written to %s', json_path)
        return log_path


def audit_require_csv(path, label):
    """Load a required CSV or exit with a clear message."""
    if not os.path.exists(path):
        print(f"FATAL: {label} not found at '{path}'.")
        print("Generate it first with the corresponding export script (see README).")
        sys.exit(1)
    import pandas as pd
    return pd.read_csv(path)
