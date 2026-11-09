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


# -- full-band audio path ----------------------------------------------------------------------

import array
import queue
import time

import talk_discord


def test_discord_to_mono48_averages_channels_without_rate_change():
    stereo = array.array("h", [100, 300, -50, 50, 7]).tobytes()
    out, rest = talk_discord.discord_to_mono48(stereo)
    assert array.array("h", out).tolist() == [200, 0]
    assert rest == array.array("h", [7]).tobytes()
    out2, rest2 = talk_discord.discord_to_mono48(array.array("h", [9]).tobytes(), carry=rest)
    assert array.array("h", out2).tolist() == [8] and rest2 == b""


def test_native_playback_passes_48k_stereo_through_untouched():
    audio = talk_discord.DiscordAudio(1, native_format=True)
    pcm = bytes(range(256)) * 15  # 3840 bytes
    audio.queue_playback(pcm)
    assert audio._outbound.get_nowait() == pcm


def test_native_capture_paces_and_pads_at_48k():
    audio = talk_discord.DiscordAudio(1, native_format=True)
    assert len(audio._input_silence) == 1920
    assert audio._input_bytes_per_second == 96_000


def test_jitter_buffer_holds_until_primed_then_plays_and_rearms():
    frames = queue.Queue()
    source = talk_discord._RealtimeSource(frames)
    source.prebuffer_bytes = 3 * talk_discord.DISCORD_FRAME_BYTES
    source.prebuffer_max_wait_s = 10
    a = b"\x01" * talk_discord.DISCORD_FRAME_BYTES
    frames.put(a)
    assert source.read() == talk_discord.SILENCE_FRAME  # 1 of 3 buffered: hold
    frames.put(a); frames.put(a)
    assert source.read() == a  # primed
    assert source.read() == a and source.read() == a
    assert source.read() == talk_discord.SILENCE_FRAME  # underrun re-arms
    frames.put(a)
    assert source.read() == talk_discord.SILENCE_FRAME


def test_jitter_buffer_flushes_a_short_tail_after_max_wait():
    frames = queue.Queue()
    source = talk_discord._RealtimeSource(frames)
    source.prebuffer_bytes = 4 * talk_discord.DISCORD_FRAME_BYTES
    source.prebuffer_max_wait_s = 0.01
    frames.put(b"\x02" * 100)
    assert source.read() == talk_discord.SILENCE_FRAME
    time.sleep(0.02)
    frame = source.read()
    assert frame[:100] == b"\x02" * 100 and len(frame) == talk_discord.DISCORD_FRAME_BYTES


def test_default_source_has_no_jitter_buffer():
    frames = queue.Queue()
    source = talk_discord._RealtimeSource(frames)
    a = b"\x03" * talk_discord.DISCORD_FRAME_BYTES
    frames.put(a)
    assert source.read() == a


# -- one-command join / leave --------------------------------------------------------------------

import asyncio as _asyncio


class _FakeRunner:
    def __init__(self, bridge_after_join=True):
        self.joined = []
        self.left = []
        self.bridge_after_join = bridge_after_join

    async def _handle_voice_channel_join(self, event):
        self.joined.append(event)
        return "Joined voice channel **General**."

    async def _handle_voice_channel_leave(self, event):
        self.left.append(event)
        return "Left voice channel."


def _wire(monkeypatch, runner, *, in_voice):
    state = {"in_voice": in_voice}
    monkeypatch.setattr(live, "_gateway_runner", lambda: runner)
    monkeypatch.setattr(live, "_voice_event", lambda r: (object(), 42))
    monkeypatch.setattr(live, "_bot_in_voice", lambda gid: state["in_voice"])
    started = []
    monkeypatch.setattr(live, "start", lambda op, gid=None: started.append((op, gid)) or "Starting GPT-Live")

    async def join(event):
        runner.joined.append(event)
        state["in_voice"] = runner.bridge_after_join
        return "Joined voice channel **General**."

    runner._handle_voice_channel_join = join
    return started


def test_join_joins_voice_first_when_bot_is_not_in_a_channel(monkeypatch):
    runner = _FakeRunner()
    started = _wire(monkeypatch, runner, in_voice=False)
    reply = _asyncio.run(live.join_and_start("111"))
    assert len(runner.joined) == 1 and started == [(111, 42)]
    assert "Joined voice channel" in reply and "Starting GPT-Live" in reply


def test_join_skips_voice_join_when_already_connected(monkeypatch):
    runner = _FakeRunner()
    started = _wire(monkeypatch, runner, in_voice=True)
    reply = _asyncio.run(live.join_and_start("111"))
    assert runner.joined == [] and started == [(111, 42)] and reply == "Starting GPT-Live"


def test_join_reports_failure_when_operator_is_not_in_voice(monkeypatch):
    runner = _FakeRunner(bridge_after_join=False)
    started = _wire(monkeypatch, runner, in_voice=False)
    reply = _asyncio.run(live.join_and_start("111"))
    assert started == [] and "Joined" in reply  # host's own message surfaces


def test_join_refuses_non_operator_before_touching_voice(monkeypatch):
    runner = _FakeRunner()
    started = _wire(monkeypatch, runner, in_voice=False)
    assert "Only a configured voice operator" in _asyncio.run(live.join_and_start("222"))
    assert runner.joined == [] and started == []


def test_leave_hangs_up_and_leaves_the_channel(monkeypatch):
    runner = _FakeRunner()
    monkeypatch.setattr(live, "_gateway_runner", lambda: runner)
    monkeypatch.setattr(live, "_voice_event", lambda r: (object(), 42))
    reply = _asyncio.run(live.stop_and_leave())
    assert len(runner.left) == 1 and "left the voice channel" in reply
