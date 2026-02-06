"""
Background task queue for download -> ingest -> enrich workflows.
"""

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from src import ingest, scraper


QUEUE_PATH = os.getenv(
    "ARXIV_ASSISTANT_TASK_QUEUE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tasks", "task_queue.json"),
)
PDF_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_filename(title: str, fallback: str = "paper", ext: str = ".pdf") -> str:
    import re

    stem = (title or "").strip()
    if stem.lower().endswith(ext):
        stem = stem[: -len(ext)]
    stem = stem.replace("/", " ").replace("\\", " ")
    stem = re.sub(r"[\t\r\n]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in " -_().,[]")
    stem = stem.strip(" .")
    if not stem:
        stem = fallback
    if len(stem) > 150:
        stem = stem[:150].rstrip()
    return f"{stem}{ext}"


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _normalize_paper_payload(paper: Dict) -> Dict:
    payload = dict(paper or {})
    # Never persist live objects (e.g. arxiv.Result)
    payload.pop("obj", None)
    return payload


def _resolve_download_url(paper: Dict) -> Optional[str]:
    pdf_url = paper.get("pdf_url")
    if pdf_url:
        return pdf_url

    source = str(paper.get("source", "")).lower()
    entry_id = str(paper.get("entry_id", "")).strip()
    if source == "arxiv" and entry_id:
        # entry_id usually like http://arxiv.org/abs/1234.5678v1
        parts = entry_id.rstrip("/").split("/")
        if parts:
            arxiv_id = parts[-1]
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return None


class TaskQueue:
    def __init__(self, queue_path: str = QUEUE_PATH):
        self.queue_path = queue_path
        self._lock = threading.RLock()
        self._tasks: List[Dict] = []
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._load()

    # ---------- Persistence ----------
    def _load(self) -> None:
        with self._lock:
            if os.path.exists(self.queue_path):
                try:
                    with open(self.queue_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._tasks = data.get("tasks", []) if isinstance(data, dict) else []
                except Exception:
                    self._tasks = []
            else:
                self._tasks = []

            # Recover interrupted runs
            changed = False
            for t in self._tasks:
                if t.get("status") == STATUS_RUNNING:
                    t["status"] = STATUS_PENDING
                    t["message"] = "Recovered from interrupted run; returned to pending."
                    changed = True
            if changed:
                self._save_locked()

    def _save_locked(self) -> None:
        _ensure_parent_dir(self.queue_path)
        tmp_path = self.queue_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"tasks": self._tasks}, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, self.queue_path)

    def _save(self) -> None:
        with self._lock:
            self._save_locked()

    # ---------- Queue control ----------
    def ensure_worker(self) -> None:
        with self._lock:
            if self._worker_thread and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._run_loop, daemon=True, name="arxiv-task-worker")
            self._worker_thread.start()

    def stop_worker(self) -> None:
        self._stop_event.set()

    def enqueue_ingest_from_paper(self, paper: Dict, run_enrichment: bool = True) -> str:
        payload = _normalize_paper_payload(paper)
        task = {
            "id": str(uuid.uuid4()),
            "type": "ingest_from_search",
            "status": STATUS_PENDING,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "message": "Queued.",
            "retries": 0,
            "cancel_requested": False,
            "payload": {
                "paper": payload,
                "download_url": _resolve_download_url(payload),
                "run_enrichment": bool(run_enrichment),
            },
        }
        with self._lock:
            self._tasks.append(task)
            self._save_locked()
        self.ensure_worker()
        return task["id"]

    def enqueue_local_file(self, pdf_path: str, metadata: Dict, run_enrichment: bool = True) -> str:
        task = {
            "id": str(uuid.uuid4()),
            "type": "ingest_local_file",
            "status": STATUS_PENDING,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "message": "Queued.",
            "retries": 0,
            "cancel_requested": False,
            "payload": {
                "pdf_path": pdf_path,
                "metadata": _normalize_paper_payload(metadata),
                "run_enrichment": bool(run_enrichment),
            },
        }
        with self._lock:
            self._tasks.append(task)
            self._save_locked()
        self.ensure_worker()
        return task["id"]

    def list_tasks(self, limit: int = 200) -> List[Dict]:
        with self._lock:
            tasks = list(self._tasks)
        tasks_sorted = sorted(tasks, key=lambda x: x.get("created_at", ""), reverse=True)
        return tasks_sorted[: max(1, int(limit))]

    def get_summary(self) -> Dict[str, int]:
        counts = {
            STATUS_PENDING: 0,
            STATUS_RUNNING: 0,
            STATUS_COMPLETED: 0,
            STATUS_FAILED: 0,
            STATUS_CANCELLED: 0,
        }
        with self._lock:
            for t in self._tasks:
                st = t.get("status")
                if st in counts:
                    counts[st] += 1
        return counts

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            for t in self._tasks:
                if t.get("id") != task_id:
                    continue
                if t.get("status") in TERMINAL_STATUSES:
                    return False
                t["cancel_requested"] = True
                if t.get("status") == STATUS_PENDING:
                    t["status"] = STATUS_CANCELLED
                    t["message"] = "Cancelled before execution."
                    t["updated_at"] = _utc_now()
                else:
                    t["message"] = "Cancellation requested."
                    t["updated_at"] = _utc_now()
                self._save_locked()
                return True
        return False

    def retry_task(self, task_id: str) -> Optional[str]:
        with self._lock:
            original = None
            for t in self._tasks:
                if t.get("id") == task_id:
                    original = t
                    break
            if not original:
                return None
            new_task = {
                "id": str(uuid.uuid4()),
                "type": original.get("type"),
                "status": STATUS_PENDING,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "message": f"Retry queued for task {task_id}.",
                "retries": int(original.get("retries", 0)) + 1,
                "cancel_requested": False,
                "payload": dict(original.get("payload", {})),
            }
            self._tasks.append(new_task)
            self._save_locked()
        self.ensure_worker()
        return new_task["id"]

    def cancel_all_pending(self) -> int:
        cancelled = 0
        with self._lock:
            for t in self._tasks:
                if t.get("status") != STATUS_PENDING:
                    continue
                if t.get("cancel_requested"):
                    continue
                t["cancel_requested"] = True
                t["status"] = STATUS_CANCELLED
                t["message"] = "Cancelled in bulk operation."
                t["updated_at"] = _utc_now()
                cancelled += 1
            if cancelled > 0:
                self._save_locked()
        return cancelled

    def retry_all_failed(self, include_cancelled: bool = False) -> int:
        with self._lock:
            retry_source_status = {STATUS_FAILED}
            if include_cancelled:
                retry_source_status.add(STATUS_CANCELLED)

            new_tasks = []
            for t in self._tasks:
                if t.get("status") not in retry_source_status:
                    continue
                new_tasks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "type": t.get("type"),
                        "status": STATUS_PENDING,
                        "created_at": _utc_now(),
                        "updated_at": _utc_now(),
                        "message": f"Bulk retry from task {t.get('id', '')[:8]}.",
                        "retries": int(t.get("retries", 0)) + 1,
                        "cancel_requested": False,
                        "payload": dict(t.get("payload", {})),
                    }
                )
            if new_tasks:
                self._tasks.extend(new_tasks)
                self._save_locked()
        if new_tasks:
            self.ensure_worker()
        return len(new_tasks)

    def prune_terminal(self, keep_latest: int = 200) -> int:
        keep_latest = max(10, int(keep_latest))
        with self._lock:
            non_terminal = [t for t in self._tasks if t.get("status") not in TERMINAL_STATUSES]
            terminal = [t for t in self._tasks if t.get("status") in TERMINAL_STATUSES]
            terminal.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            kept_terminal = terminal[:keep_latest]
            removed = len(terminal) - len(kept_terminal)
            self._tasks = non_terminal + kept_terminal
            if removed > 0:
                self._save_locked()
        return max(0, removed)

    # ---------- Worker ----------
    def _next_pending_task(self) -> Optional[Dict]:
        with self._lock:
            pending = [t for t in self._tasks if t.get("status") == STATUS_PENDING and not t.get("cancel_requested")]
            if not pending:
                return None
            pending.sort(key=lambda x: x.get("created_at", ""))
            task = pending[0]
            task["status"] = STATUS_RUNNING
            task["updated_at"] = _utc_now()
            task["started_at"] = _utc_now()
            task["message"] = "Running."
            self._save_locked()
            return dict(task)

    def _update_task(self, task_id: str, **kwargs) -> None:
        with self._lock:
            for t in self._tasks:
                if t.get("id") == task_id:
                    t.update(kwargs)
                    t["updated_at"] = _utc_now()
                    self._save_locked()
                    return

    def _is_cancel_requested(self, task_id: str) -> bool:
        with self._lock:
            for t in self._tasks:
                if t.get("id") == task_id:
                    return bool(t.get("cancel_requested"))
        return False

    def _finalize_task(self, task_id: str, status: str, message: str) -> None:
        self._update_task(task_id, status=status, message=message, finished_at=_utc_now())

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self._next_pending_task()
            if not task:
                time.sleep(0.7)
                continue

            task_id = task.get("id")
            try:
                if self._is_cancel_requested(task_id):
                    self._finalize_task(task_id, STATUS_CANCELLED, "Cancelled before execution.")
                    continue

                task_type = task.get("type")
                if task_type == "ingest_from_search":
                    self._run_ingest_from_search(task_id, task.get("payload", {}))
                elif task_type == "ingest_local_file":
                    self._run_ingest_local_file(task_id, task.get("payload", {}))
                else:
                    self._finalize_task(task_id, STATUS_FAILED, f"Unsupported task type: {task_type}")
            except Exception as e:
                self._finalize_task(task_id, STATUS_FAILED, f"Unhandled error: {str(e)}")

    # ---------- Task handlers ----------
    def _run_ingest_from_search(self, task_id: str, payload: Dict) -> None:
        paper = dict(payload.get("paper") or {})
        title = paper.get("title", "paper")
        download_url = payload.get("download_url")
        run_enrichment = bool(payload.get("run_enrichment", True))

        if self._is_cancel_requested(task_id):
            self._finalize_task(task_id, STATUS_CANCELLED, "Cancelled.")
            return

        if not download_url:
            self._finalize_task(task_id, STATUS_FAILED, "No downloadable PDF URL available.")
            return

        self._update_task(task_id, message="Downloading PDF...")
        pdf_path = scraper.download_from_url(download_url, title, PDF_DIR)
        if not pdf_path:
            self._finalize_task(task_id, STATUS_FAILED, "Download failed.")
            return

        if self._is_cancel_requested(task_id):
            self._finalize_task(task_id, STATUS_CANCELLED, "Cancelled after download.")
            return

        self._update_task(task_id, message="Ingesting PDF and running incremental pipeline...")
        ingest_ok = ingest.ingest_paper(
            pdf_path,
            paper,
            run_incremental_pipeline=True,
            run_metadata_fix=run_enrichment,
        )
        if not ingest_ok:
            self._finalize_task(task_id, STATUS_FAILED, "Ingest failed.")
            return

        self._finalize_task(task_id, STATUS_COMPLETED, f"Completed: {title}")

    def _run_ingest_local_file(self, task_id: str, payload: Dict) -> None:
        pdf_path = payload.get("pdf_path")
        metadata = dict(payload.get("metadata") or {})
        title = metadata.get("title") or os.path.basename(str(pdf_path or "")) or "paper"
        run_enrichment = bool(payload.get("run_enrichment", True))

        if not pdf_path or not os.path.exists(str(pdf_path)):
            self._finalize_task(task_id, STATUS_FAILED, f"Local file not found: {pdf_path}")
            return

        if self._is_cancel_requested(task_id):
            self._finalize_task(task_id, STATUS_CANCELLED, "Cancelled.")
            return

        self._update_task(task_id, message="Ingesting local PDF and running incremental pipeline...")
        ingest_ok = ingest.ingest_paper(
            pdf_path,
            metadata,
            run_incremental_pipeline=True,
            run_metadata_fix=run_enrichment,
        )
        if not ingest_ok:
            self._finalize_task(task_id, STATUS_FAILED, "Ingest failed.")
            return

        self._finalize_task(task_id, STATUS_COMPLETED, f"Completed: {title}")


_queue_instance: Optional[TaskQueue] = None
_queue_lock = threading.Lock()


def get_queue() -> TaskQueue:
    global _queue_instance
    with _queue_lock:
        if _queue_instance is None:
            _queue_instance = TaskQueue()
        _queue_instance.ensure_worker()
        return _queue_instance
