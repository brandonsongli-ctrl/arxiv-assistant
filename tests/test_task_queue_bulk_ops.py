from src import task_queue


def _task(task_id, status, created_at):
    return {
        "id": task_id,
        "type": "ingest_from_search",
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "message": "seed",
        "retries": 0,
        "cancel_requested": False,
        "payload": {"paper": {"title": f"Paper {task_id}"}},
    }


def test_bulk_cancel_retry_and_prune(tmp_path):
    queue_path = tmp_path / "task_queue.json"
    q = task_queue.TaskQueue(queue_path=str(queue_path))
    q.ensure_worker = lambda: None

    completed_seed = [
        _task(f"done-{i}", task_queue.STATUS_COMPLETED, f"2026-01-01T00:00:{10 + i:02d}Z")
        for i in range(12)
    ]
    q._tasks = [
        _task("pending-1", task_queue.STATUS_PENDING, "2026-01-01T00:00:00Z"),
        _task("pending-2", task_queue.STATUS_PENDING, "2026-01-01T00:00:01Z"),
        _task("failed-1", task_queue.STATUS_FAILED, "2026-01-01T00:00:02Z"),
        _task("cancelled-1", task_queue.STATUS_CANCELLED, "2026-01-01T00:00:03Z"),
    ] + completed_seed
    q._save()

    cancelled = q.cancel_all_pending()
    assert cancelled == 2
    assert all(t["status"] != task_queue.STATUS_PENDING for t in q._tasks)

    retried_failed_only = q.retry_all_failed(include_cancelled=False)
    assert retried_failed_only == 1
    pending_after_failed_retry = [t for t in q._tasks if t["status"] == task_queue.STATUS_PENDING]
    assert len(pending_after_failed_retry) == 1

    retried_with_cancelled = q.retry_all_failed(include_cancelled=True)
    assert retried_with_cancelled == 4
    pending_all = [t for t in q._tasks if t["status"] == task_queue.STATUS_PENDING]
    assert len(pending_all) == 5

    removed = q.prune_terminal(keep_latest=2)
    assert removed > 0
    terminal_left = [t for t in q._tasks if t["status"] in task_queue.TERMINAL_STATUSES]
    assert len(terminal_left) == 10
