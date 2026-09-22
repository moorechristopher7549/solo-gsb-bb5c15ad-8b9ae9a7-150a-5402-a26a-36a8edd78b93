"""POST /events/{event_id}/extend 受约束截止时间延长测试。

SQLite 文件库 + 模拟时钟（与其余测试相同手法），覆盖：
  - commit 阶段延长成功：返回旧值/新值/reason，事件截止时间随之更新
  - 只能延长：新 commit 不晚于旧值、新 reveal 不晚于旧值/新 commit -> 422，且状态不变
  - reveal / closed 阶段 -> 409，且状态不变
  - 时间无时区 / reason 缺失 -> 422（Pydantic）
  - 未知事件 -> 404
  - 完全相同的重复提交幂等：返回与首次相同的旧值/新值/reason/extended_at
  - 延长后仍可在新窗口内提交承诺、旧的迟到时刻变为可提交
  - 失败事务不留下 deadline_extensions 记录
  - 延长记录进入可核验时间线
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from uuid import UUID

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_extend_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/extend.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.main as main_module  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402
from app.models import DeadlineExtension  # noqa: E402

T0 = datetime(2030, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)
NEW_COMMIT_DL = T0 + timedelta(hours=3)
NEW_REVEAL_DL = T0 + timedelta(hours=4)
REASON = "评审需要更多收集时间"


def iso(dt: datetime) -> str:
    """FastAPI 把 UTC 时间序列化为 ...Z（而非 +00:00），断言响应时统一形态。"""
    s = dt.astimezone(timezone.utc).isoformat()
    if s.endswith("+00:00"):
        s = s[: -len("+00:00")] + "Z"
    return s


class MutableClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


clock = MutableClock(T0)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    monkeypatch.setattr(main_module, "now_utc", clock)
    yield


client = TestClient(main_module.app)


def extension_count(event_id: str) -> int:
    db = SessionLocal()
    try:
        return (
            db.query(DeadlineExtension)
            .filter(DeadlineExtension.event_id == UUID(event_id))
            .count()
        )
    finally:
        db.close()


@pytest.fixture
def event_id() -> str:
    with client:
        r = client.post("/events", json={
            "title": "延长测试事件",
            "commit_deadline": COMMIT_DL.isoformat(),
            "reveal_deadline": REVEAL_DL.isoformat(),
        })
        assert r.status_code == 201, r.text
        return r.json()["id"]


def extend(event_id: str, commit_dl: datetime, reveal_dl: datetime, reason: str = REASON):
    return client.post(
        f"/events/{event_id}/extend",
        json={
            "new_commit_deadline": commit_dl.isoformat(),
            "new_reveal_deadline": reveal_dl.isoformat(),
            "reason": reason,
        },
    )


def test_extend_success_returns_old_and_new(event_id):
    r = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["event_id"] == event_id
    assert body["reason"] == REASON
    assert body["old_commit_deadline"] == iso(COMMIT_DL)
    assert body["new_commit_deadline"] == iso(NEW_COMMIT_DL)
    assert body["old_reveal_deadline"] == iso(REVEAL_DL)
    assert body["new_reveal_deadline"] == iso(NEW_REVEAL_DL)
    assert "extended_at" in body

    # 事件本身的截止时间已在同一事务中更新，状态仍为 commit
    ev = client.get(f"/events/{event_id}").json()
    assert ev["commit_deadline"] == iso(NEW_COMMIT_DL)
    assert ev["reveal_deadline"] == iso(NEW_REVEAL_DL)
    assert ev["status"] == "commit"
    assert extension_count(event_id) == 1


def test_extend_accepts_timezone_offset_input(event_id):
    # +08:00 表示的同一时刻必须按绝对时间比较（UTC = 03:00Z / 04:00Z）
    tz8 = timezone(timedelta(hours=8))
    r = extend(
        event_id,
        NEW_COMMIT_DL.astimezone(tz8),
        NEW_REVEAL_DL.astimezone(tz8),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["new_commit_deadline"] == iso(NEW_COMMIT_DL)
    assert body["new_reveal_deadline"] == iso(NEW_REVEAL_DL)
    assert client.get(f"/events/{event_id}").json()["commit_deadline"] == iso(NEW_COMMIT_DL)


def test_extend_idempotent_same_request(event_id):
    r1 = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r1.status_code == 200, r1.text
    first = r1.json()

    # 时钟推进后重放完全相同的请求，必须逐字段返回同一结果
    clock.t = T0 + timedelta(minutes=30)
    r2 = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r2.status_code == 200, r2.text
    assert r2.json() == first
    assert extension_count(event_id) == 1  # 不产生第二条记录
    assert client.get(f"/events/{event_id}").json()["commit_deadline"] == iso(NEW_COMMIT_DL)


def test_extend_idempotent_after_leaving_commit_phase(event_id):
    # 首次成功延长到 3h/4h
    first = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL).json()
    # 推进到 reveal 阶段后重放同一请求：幂等重放仍返回首次结果，而非 409
    clock.t = NEW_COMMIT_DL
    r = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r.status_code == 200, r.text
    assert r.json() == first
    assert extension_count(event_id) == 1


def test_extend_chain_each_step_strictly_later(event_id):
    r1 = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL, reason="第一次")
    assert r1.status_code == 200, r1.text
    assert r1.json()["old_commit_deadline"] == iso(COMMIT_DL)

    later_commit = T0 + timedelta(hours=5)
    later_reveal = T0 + timedelta(hours=6)
    r2 = extend(event_id, later_commit, later_reveal, reason="第二次")
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    # 第二次的旧值必须是第一次的新值
    assert body2["old_commit_deadline"] == iso(NEW_COMMIT_DL)
    assert body2["new_commit_deadline"] == iso(later_commit)
    assert body2["old_reveal_deadline"] == iso(NEW_REVEAL_DL)
    assert extension_count(event_id) == 2


def test_extend_rejects_equal_commit_deadline(event_id):
    # 新 commit == 当前 commit：不是严格延长
    r = extend(event_id, COMMIT_DL, NEW_REVEAL_DL)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "DEADLINE_NOT_EXTENDED"
    ev = client.get(f"/events/{event_id}").json()
    assert ev["commit_deadline"] == iso(COMMIT_DL)
    assert extension_count(event_id) == 0


def test_extend_rejects_shorter_commit_deadline(event_id):
    r = extend(event_id, COMMIT_DL - timedelta(minutes=1), NEW_REVEAL_DL)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "DEADLINE_NOT_EXTENDED"
    assert client.get(f"/events/{event_id}").json()["commit_deadline"] == iso(COMMIT_DL)
    assert extension_count(event_id) == 0


def test_extend_rejects_reveal_not_after_new_commit(event_id):
    # 新 reveal == 新 commit
    r = extend(event_id, NEW_COMMIT_DL, NEW_COMMIT_DL)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_DEADLINE_ORDER"
    assert client.get(f"/events/{event_id}").json()["commit_deadline"] == iso(COMMIT_DL)

    # 新 reveal < 新 commit
    r = extend(event_id, NEW_REVEAL_DL, NEW_COMMIT_DL)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "INVALID_DEADLINE_ORDER"
    assert extension_count(event_id) == 0


def test_extend_rejects_shorter_reveal_even_if_after_new_commit(event_id):
    # 新 commit 只延长到 90 分钟；新 reveal = 100 分钟虽然晚于新 commit，
    # 却早于当前 reveal（120 分钟）：reveal 同样只能延长不能缩短
    r = extend(
        event_id,
        T0 + timedelta(minutes=90),
        T0 + timedelta(minutes=100),
    )
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "DEADLINE_NOT_EXTENDED"
    ev = client.get(f"/events/{event_id}").json()
    assert ev["commit_deadline"] == iso(COMMIT_DL)
    assert ev["reveal_deadline"] == iso(REVEAL_DL)
    assert extension_count(event_id) == 0


def test_extend_in_reveal_phase_returns_409(event_id):
    clock.t = COMMIT_DL  # 进入 reveal
    r = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "DEADLINE_EXTENSION_PHASE_CLOSED"
    ev = client.get(f"/events/{event_id}").json()
    assert ev["status"] == "reveal"
    assert ev["commit_deadline"] == iso(COMMIT_DL)
    assert extension_count(event_id) == 0


def test_extend_in_closed_phase_returns_409(event_id):
    clock.t = REVEAL_DL  # closed
    r = extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "DEADLINE_EXTENSION_PHASE_CLOSED"
    assert client.get(f"/events/{event_id}").json()["reveal_deadline"] == iso(REVEAL_DL)
    assert extension_count(event_id) == 0


def test_extend_naive_datetime_returns_422(event_id):
    r = client.post(f"/events/{event_id}/extend", json={
        "new_commit_deadline": "2030-05-01T03:00:00",  # 无 Z / 偏移
        "new_reveal_deadline": NEW_REVEAL_DL.isoformat(),
        "reason": REASON,
    })
    assert r.status_code == 422, r.text
    assert client.get(f"/events/{event_id}").json()["commit_deadline"] == iso(COMMIT_DL)

    # commit 带时区、reveal 不带 -> 同样 422
    r = client.post(f"/events/{event_id}/extend", json={
        "new_commit_deadline": NEW_COMMIT_DL.isoformat(),
        "new_reveal_deadline": "2030-05-01T04:00:00",
        "reason": REASON,
    })
    assert r.status_code == 422
    assert extension_count(event_id) == 0


def test_extend_missing_or_empty_reason_returns_422(event_id):
    r = client.post(f"/events/{event_id}/extend", json={
        "new_commit_deadline": NEW_COMMIT_DL.isoformat(),
        "new_reveal_deadline": NEW_REVEAL_DL.isoformat(),
    })
    assert r.status_code == 422
    r = client.post(f"/events/{event_id}/extend", json={
        "new_commit_deadline": NEW_COMMIT_DL.isoformat(),
        "new_reveal_deadline": NEW_REVEAL_DL.isoformat(),
        "reason": "   ",
    })
    assert r.status_code == 422
    assert extension_count(event_id) == 0


def test_extend_unknown_event_returns_404():
    missing = "00000000-0000-0000-0000-000000000000"
    r = extend(missing, NEW_COMMIT_DL, NEW_REVEAL_DL)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "EVENT_NOT_FOUND"


def test_extend_non_uuid_returns_422():
    r = client.post("/events/not-a-uuid/extend", json={
        "new_commit_deadline": NEW_COMMIT_DL.isoformat(),
        "new_reveal_deadline": NEW_REVEAL_DL.isoformat(),
        "reason": REASON,
    })
    assert r.status_code == 422


def test_extend_opens_new_commit_window(event_id):
    # 先在旧 commit 窗口内提交一个承诺
    h1 = compute_commitment("pred-1", "salt-1")
    assert client.post(
        f"/events/{event_id}/commitments", json={"commitment_hash": h1}
    ).status_code == 201

    # 尚在 commit 阶段时延长到 3h/4h
    assert extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL).status_code == 200

    # 旧的 commit_deadline 时刻：未经延长本应迟到，现在仍可提交
    clock.t = COMMIT_DL
    ev = client.get(f"/events/{event_id}").json()
    assert ev["status"] == "commit"
    assert client.post(
        f"/events/{event_id}/commitments",
        json={"commitment_hash": compute_commitment("late", "s")},
    ).status_code == 201

    # 到达新 commit_deadline 才关闭提交、开启揭示
    clock.t = NEW_COMMIT_DL
    ev = client.get(f"/events/{event_id}").json()
    assert ev["status"] == "reveal"
    assert client.post(
        f"/events/{event_id}/commitments",
        json={"commitment_hash": compute_commitment("late2", "s")},
    ).status_code == 409


def test_extend_records_appear_in_timeline(event_id):
    extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL, reason="时间线原因")
    r = client.get(f"/events/{event_id}/timeline")
    assert r.status_code == 200, r.text
    entries = r.json()["entries"]
    ext_entries = [e for e in entries if e["type"] == "deadline_extended"]
    assert len(ext_entries) == 1
    detail = ext_entries[0]["detail"]
    # 时间线 detail 内沿用既有约定：裸 isoformat（+00:00）
    assert detail["old_commit_deadline"] == COMMIT_DL.isoformat()
    assert detail["new_commit_deadline"] == NEW_COMMIT_DL.isoformat()
    assert detail["old_reveal_deadline"] == REVEAL_DL.isoformat()
    assert detail["new_reveal_deadline"] == NEW_REVEAL_DL.isoformat()
    assert detail["reason"] == "时间线原因"
    # seq 连续
    assert [e["seq"] for e in entries] == list(range(1, len(entries) + 1))


def test_failed_extension_leaves_no_audit_trail(event_id):
    # 两次失败（阶段外 + 约束非法），一次成功 -> 仅 1 条审计记录
    clock.t = REVEAL_DL
    assert extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL).status_code == 409
    clock.t = T0
    assert extend(event_id, COMMIT_DL, NEW_REVEAL_DL).status_code == 422
    assert extend(event_id, NEW_COMMIT_DL, NEW_REVEAL_DL).status_code == 200
    assert extension_count(event_id) == 1
