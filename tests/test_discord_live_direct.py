"""Local fork lane: GPT-Live on Discord without host task context (talk_discord_live)."""

from types import SimpleNamespace as N

import pytest

import talk_discord_live as live


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    live.reset_for_tests()
    monkeypatch.setenv("TALK_DISCORD_OPERATOR_USER_IDS", "111")
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "111,222")
    monkeypatch.delenv("TALK_DISCORD_AUDIENCE_USER_IDS", raising=False)
    yield
    live.reset_for_tests()


def test_operator_speech_passes_and_others_become_equal_length_silence():
    ops = frozenset({111})
    assert live.admit_packet(N(speaker={"user_id": "111"}, pcm=b"\x01\x02"), ops) == b"\x01\x02"
    assert live.admit_packet(N(speaker={"user_id": 222}, pcm=b"\x01\x02"), ops) == b"\x00\x00"
    assert live.admit_packet(N(speaker={"user_id": None}, pcm=b"\x01\x02"), ops) == b"\x00\x00"
    assert live.admit_packet(N(speaker=None, pcm=b"\x05\x06"), ops) == b"\x05\x06"
    assert live.admit_packet(None, ops) is None


def test_room_policy_allows_allowlisted_and_bots_only():
    allowed = live.audience_ids()
    assert allowed == frozenset({111, 222})
    members = [N(id=111, bot=False), N(id=222, bot=False), N(id=5, bot=True)]
    vc = N(channel=N(members=members), is_connected=lambda: True)
    assert live.room_violation(vc, allowed) is None
    members.append(N(id=777, bot=False, display_name="Rando"))
    assert "Rando" in live.room_violation(vc, allowed)
    assert live.room_violation(N(channel=None, is_connected=lambda: True), allowed)
    assert live.room_violation(N(channel=N(members=[]), is_connected=lambda: False), allowed)


def test_ledger_matches_hook_payloads_by_prompt_text():
    turn = live._PendingTurn("d1", "Check the weather in SF", "check the weather in sf", "CTX")
    live._LEDGER.add(turn)
    assert live._LEDGER.context_for("[voice] Check the   weather in SF") == "CTX"
    assert live._LEDGER.context_for("something else") is None
    resolved = live._LEDGER.resolve([{"type": "text", "text": "check the weather in SF"}], "61F")
    assert resolved is turn and turn.result == "61F" and turn.done.is_set()
    assert live._LEDGER.context_for("Check the weather in SF") is None  # done turns never rematch


def test_hooks_are_inert_without_a_session():
    live._LEDGER.add(live._PendingTurn("d1", "hello there", "hello there", "CTX"))
    assert live.pre_llm_call_hook(user_message="hello there") is None
    assert live.post_llm_call_hook(user_message="hello there", assistant_response="x") is None


def test_start_refuses_non_operators_and_missing_config(monkeypatch):
    import asyncio

    async def start(user):
        return live.start(user)

    assert "Only a configured voice operator" in asyncio.run(start("222"))
    monkeypatch.setenv("TALK_DISCORD_OPERATOR_USER_IDS", "")
    monkeypatch.setattr(live.talk_config, "discord_operator_user_ids", lambda: frozenset())
    assert "TALK_DISCORD_OPERATOR_USER_IDS" in asyncio.run(start("111"))


def test_start_needs_gateway_loop():
    assert "event loop" in live.start("111")
