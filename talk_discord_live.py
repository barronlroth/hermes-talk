"""GPT-Live (``gpt-live-1-codex``) in a Discord voice channel, without host task-context support.

Local fork lane ("live-direct"). Upstream Hermes Talk refuses GPT-Live on Discord unless the host
ships ``gateway.discord_task_context`` (a Talk-enabled Hermes build). This lane keeps the same
audio bridge (:class:`talk_discord.DiscordAudio`) and the same subscription transport
(:class:`talk_live_realtime.LiveRealtimeSession`), and replaces the host's room-authority proof
with a policy the plugin enforces itself:

* Only operator speech (``TALK_DISCORD_OPERATOR_USER_IDS``) reaches the model. Anyone else's
  audio is replaced with silence of the same length, so pacing and turn detection stay intact.
* Everyone in the voice channel must be an operator or on ``DISCORD_ALLOWED_USERS`` (bots
  excluded). Anyone else joining ends the session at once: playback is dropped first.
* GPT-Live is only the voice. Each client delegation becomes an ordinary Hermes turn in the
  Discord text channel bound by ``/voice join``, sent as the operator through the gateway's own
  voice-input path (authorization, channel prompt, transcript echo). The turn's final reply is
  captured by a ``post_llm_call`` hook and handed back to GPT-Live to speak.
* Billing stays fail-closed: subscription auth only, never an API key.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

try:
    from . import talk_config, talk_discord
    from . import talk_realtime as rt
    from .talk_live_config import resolve_live_auth, resolve_live_config
    from .talk_live_realtime import LiveRealtimeSession
except ImportError:  # pragma: no cover - flat-module fallback
    import talk_config
    import talk_discord
    import talk_realtime as rt
    from talk_live_config import resolve_live_auth, resolve_live_config
    from talk_live_realtime import LiveRealtimeSession

_log = logging.getLogger(__name__)

#: How long a delegated Hermes turn may run before GPT-Live tells the operator it will finish in text.
DELEGATION_SPOKEN_TIMEOUT_S = float(os.environ.get("TALK_LIVE_DISCORD_TURN_TIMEOUT_S") or 120)
#: A late result is still spoken if it lands within this window after the spoken timeout.
LATE_RESULT_WINDOW_S = 900.0
AUDIENCE_CHECK_INTERVAL_S = 1.0
INPUT_POLL_S = 0.005
RECENT_TRANSCRIPT_LINES = 12

LIVE_PERSONA = (
    "You are {name}, the user's personal agent, speaking in a Discord voice channel. You are the "
    "voice layer only. Speak naturally and casually, like a sharp friend: short sentences, no "
    "corporate filler. Keep replies brief unless asked for detail.\n\n"
    "Backchannel policy: Use light backchannels. Acknowledge naturally without competing with "
    "the main response.\n\n"
    "Interruption policy: Stop speaking when the user interrupts. Listen to what they say.\n\n"
    "Delegation policy:\n"
    "Backend tools:\n"
    "- {name} backend: a full AI agent with tools and memory. It can check calendars, email, "
    "messages, the web, files, finances, reminders and smart home, run commands, remember "
    "things, and reason carefully. It is the one who actually does work and knows facts.\n\n"
    "Delegate to the backend when:\n"
    "- The user asks a question that needs facts, personal context, current information, or "
    "careful reasoning.\n"
    "- The user asks you to do, check, find, make, fix, send, run or remember anything.\n"
    "- A correction changes work already requested.\n\n"
    "Do not delegate to the backend when:\n"
    "- The user greets you, makes small talk, or asks you to repeat a result already provided.\n"
    "- You need a brief clarification to understand the request.\n\n"
    "Delegate before giving an answer that depends on backend work. Never guess or invent the "
    "result while waiting; say briefly that you are checking, then wait for the result. When a "
    "result arrives, say it conversationally and briefly."
)

VOICE_TURN_NOTE = (
    "[Note: this message is a delegation from a live spoken conversation in a Discord voice "
    "channel (GPT-Live voice layer). The text is a voice transcript and may contain mis-hearings "
    "and corrections; use the latest intent. Your final reply will be spoken aloud by the voice "
    "model: answer in plain conversational sentences, keep it short (two or three sentences "
    "unless the user asked for detail), no markdown, no lists, no code blocks, no URLs. Do the "
    "work with your tools as usual. Never claim an action succeeded before it actually did.]"
)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


@dataclass
class _PendingTurn:
    delegation_id: str
    prompt: str
    key: str
    context: str
    created: float = field(default_factory=time.monotonic)
    result: str | None = None
    done: threading.Event = field(default_factory=threading.Event)
    spoken_timeout: bool = False


class _TurnLedger:
    """Delegations waiting on a Hermes turn. Matched by prompt text inside the hook payloads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[_PendingTurn] = []

    def add(self, turn: _PendingTurn) -> None:
        with self._lock:
            self._items.append(turn)

    def _match(self, user_message: Any) -> _PendingTurn | None:
        text = _norm(user_message if isinstance(user_message, str) else _flatten(user_message))
        if not text:
            return None
        for turn in self._items:
            if not turn.done.is_set() and turn.key and turn.key in text:
                return turn
        return None

    def context_for(self, user_message: Any) -> str | None:
        with self._lock:
            turn = self._match(user_message)
            return turn.context if turn else None

    def resolve(self, user_message: Any, response: Any) -> _PendingTurn | None:
        with self._lock:
            turn = self._match(user_message)
            if turn is None:
                return None
            turn.result = str(response or "").strip()
            turn.done.set()
            return turn

    def prune(self) -> None:
        cutoff = time.monotonic() - LATE_RESULT_WINDOW_S - DELEGATION_SPOKEN_TIMEOUT_S
        with self._lock:
            self._items = [t for t in self._items if t.created > cutoff and not t.done.is_set()]

    def clear(self) -> None:
        with self._lock:
            for turn in self._items:
                turn.done.set()
            self._items.clear()


def _flatten(message: Any) -> str:
    if isinstance(message, list):
        parts = []
        for part in message:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return " ".join(parts)
    return str(message or "")


_LEDGER = _TurnLedger()
_STATE_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


# ---------------------------------------------------------------------------------------------
# Hooks (registered by the plugin at load time; inert unless a live-direct session is running)
# ---------------------------------------------------------------------------------------------

def pre_llm_call_hook(**kwargs: Any) -> dict | None:
    if not session_active():
        return None
    context = _LEDGER.context_for(kwargs.get("user_message"))
    return {"context": context} if context else None


def post_llm_call_hook(**kwargs: Any) -> None:
    if not session_active():
        return
    turn = _LEDGER.resolve(kwargs.get("user_message"), kwargs.get("assistant_response"))
    if turn is None:
        return
    with _STATE_LOCK:
        loop = _STATE.get("loop")
        notify = _STATE.get("on_result")
    if loop is not None and notify is not None:
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(notify, turn)


# ---------------------------------------------------------------------------------------------
# Room policy
# ---------------------------------------------------------------------------------------------

def _id_set(raw: str | None) -> frozenset[int]:
    out = set()
    for piece in re.split(r"[\s,]+", raw or ""):
        if piece.isdigit():
            out.add(int(piece))
    return frozenset(out)


def operator_ids() -> frozenset[int]:
    try:
        return frozenset(talk_config.discord_operator_user_ids())
    except Exception:  # noqa: BLE001 - malformed config refuses below
        return _id_set(os.environ.get("TALK_DISCORD_OPERATOR_USER_IDS"))


def audience_ids() -> frozenset[int]:
    return operator_ids() | _id_set(os.environ.get("DISCORD_ALLOWED_USERS")) | _id_set(
        os.environ.get("TALK_DISCORD_AUDIENCE_USER_IDS")
    )


def room_violation(voice_client: Any, allowed: frozenset[int]) -> str | None:
    """Name the first reason this room may not hear the session, or None when it may."""

    channel = getattr(voice_client, "channel", None)
    if channel is None or not getattr(voice_client, "is_connected", lambda: False)():
        return "the voice connection dropped"
    members = getattr(channel, "members", None)
    if members is None:
        return "the voice channel's member list is unavailable"
    for member in members:
        if getattr(member, "bot", False):
            continue
        member_id = getattr(member, "id", None)
        if member_id not in allowed:
            name = getattr(member, "display_name", None) or str(member_id)
            return f"{name} joined the channel and is not on the voice allowlist"
    return None


def admit_packet(packet: Any, operators: frozenset[int]) -> bytes | None:
    """Operator speech passes; everyone else becomes silence of the same length (pacing intact)."""

    if packet is None:
        return None
    pcm = getattr(packet, "pcm", b"")
    if not pcm:
        return None
    speaker = getattr(packet, "speaker", None)
    if speaker is None:  # synthesized silence from the bridge
        return pcm
    user_id = speaker.get("user_id") if isinstance(speaker, dict) else None
    try:
        user_id = int(user_id) if user_id is not None else None
    except (TypeError, ValueError):
        user_id = None
    if user_id is not None and user_id in operators:
        return pcm
    return bytes(len(pcm))


# ---------------------------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------------------------

def _full_band_transport(*, auth, config):
    """GPT-Live WebRTC transport speaking Discord's native geometry: 48 kHz mono in, 48 kHz stereo out."""

    try:
        from .talk_live_audio import LiveWebRTCTransport
    except ImportError:  # pragma: no cover - flat-module fallback
        from talk_live_audio import LiveWebRTCTransport
    return LiveWebRTCTransport(
        auth=auth, config=config, input_rate=48_000, output_rate=48_000, output_layout="stereo"
    )


def session_active() -> bool:
    with _STATE_LOCK:
        task = _STATE.get("task")
        return task is not None and not task.done()


def _gateway_runner():
    from gateway.run import _gateway_runner_ref  # host-only, lazy by design

    runner = _gateway_runner_ref() if callable(_gateway_runner_ref) else None
    if runner is None:
        raise talk_discord.TalkDiscordError("the Hermes gateway isn't running in this process")
    return runner


class LiveDirectSession:
    def __init__(self, *, bridge: dict, operator_id: int, runner: Any) -> None:
        self.bridge = bridge
        self.guild_id = bridge["guild_id"]
        self.adapter = bridge["adapter"]
        self.voice_client = bridge["voice_client"]
        self.operator_id = operator_id
        self.runner = runner
        self.operators = operator_ids()
        self.allowed = audience_ids()
        # Full-band: 48 kHz in and out, no 24 kHz round trip, plus a small jitter buffer.
        self.prebuffer_ms = int(os.environ.get("TALK_LIVE_DISCORD_PREBUFFER_MS") or 80)
        self.audio = talk_discord.DiscordAudio(
            self.guild_id, native_format=True, prebuffer_ms=self.prebuffer_ms
        )
        self.session: LiveRealtimeSession | None = None
        self.transcript: list[str] = []
        self.ended_reason: str | None = None
        self.started_at = time.monotonic()
        self.delegations = 0
        self._send_lock = asyncio.Lock()

    # -- helpers -----------------------------------------------------------------------------
    def text_channel_id(self) -> int | None:
        channels = getattr(self.adapter, "_voice_text_channels", None) or {}
        return channels.get(self.guild_id)

    async def post_text(self, text: str) -> None:
        channel_id = self.text_channel_id()
        client = getattr(self.adapter, "_client", None)
        if not channel_id or client is None:
            return
        with suppress(Exception):
            channel = client.get_channel(int(channel_id))
            if channel is not None:
                await channel.send(text[:1900])

    async def send(self, commands: list) -> None:
        if self.session is None:
            return
        async with self._send_lock:
            await self.session.send(commands)

    def recent_context(self) -> str:
        return "\n".join(self.transcript[-RECENT_TRANSCRIPT_LINES:])

    # -- lifecycle -----------------------------------------------------------------------------
    async def run(self) -> None:
        violation = room_violation(self.voice_client, self.allowed)
        if violation:
            raise talk_discord.TalkDiscordError(f"not starting GPT-Live: {violation}")
        config = resolve_live_config()
        if config.auth_mode != "subscription":
            raise talk_discord.TalkDiscordError(
                "live-direct Discord voice only runs on the Codex subscription (TALK_LIVE_AUTH)"
            )
        auth = await asyncio.to_thread(resolve_live_auth, config=config)
        extra = (os.environ.get("TALK_LIVE_DISCORD_INSTRUCTIONS") or "").strip()
        name = (os.environ.get("TALK_LIVE_DISCORD_NAME") or "Hermes").strip() or "Hermes"
        persona = LIVE_PERSONA.replace("{name}", name)
        setup = rt.SessionSetup(
            model=config.model,
            voice=config.voice,
            instructions=f"{persona}\n\n{extra}" if extra else persona,
        )
        self.session = LiveRealtimeSession(
            auth=auth, config=config, transport_factory=_full_band_transport
        )
        await self.session.connect(setup)
        self.audio.start()
        _log.info("live-direct: GPT-Live connected (%s/%s) guild=%s", config.model, config.voice,
                  self.guild_id)
        await self.post_text(
            f"🎙️ GPT-Live is on (`{config.model}`, voice `{config.voice}`, Codex subscription). "
            "Just talk. `/talk leave` hangs up."
        )
        pump = asyncio.create_task(self._pump_input())
        guard = asyncio.create_task(self._guard_room())
        try:
            async for event in self.session:
                await self._handle(event)
                if self.ended_reason:
                    break
        finally:
            for task in (pump, guard):
                task.cancel()
            for task in (pump, guard):
                with suppress(BaseException):
                    await task

    async def close(self) -> None:
        with suppress(Exception):
            self.audio.drain_playback()
        with suppress(Exception):
            self.audio.stop()
        if self.session is not None:
            with suppress(Exception):
                await self.session.close()
        _LEDGER.clear()

    # -- loops ---------------------------------------------------------------------------------
    async def _pump_input(self) -> None:
        while True:
            packet = await asyncio.to_thread(self.audio.read_input_packet)
            if packet is None:
                await asyncio.sleep(INPUT_POLL_S)
                continue
            pcm = admit_packet(packet, self.operators)
            if pcm:
                await self.send([rt.AppendInputAudio(pcm)])

    async def _guard_room(self) -> None:
        while True:
            await asyncio.sleep(AUDIENCE_CHECK_INTERVAL_S)
            violation = room_violation(self.voice_client, self.allowed)
            if violation:
                self.ended_reason = violation
                with suppress(Exception):
                    self.audio.drain_playback()
                    self.audio.pause_input()
                _log.warning("live-direct: ending session, %s", violation)
                await self.post_text(f"🔇 GPT-Live hung up: {violation}.")
                if self.session is not None:
                    with suppress(Exception):
                        await self.session.close()
                return

    # -- events --------------------------------------------------------------------------------
    async def _handle(self, event: Any) -> None:
        if isinstance(event, rt.OutputAudio):
            if not self.ended_reason:
                self.audio.queue_playback(event.data)
            return
        if isinstance(event, (rt.SpeechStarted, rt.OutputInterrupted)):
            self.audio.drain_playback()
            return
        if isinstance(event, rt.Transcript):
            if event.final and event.text:
                who = "User" if event.role == rt.TranscriptRole.USER else "Assistant"
                self.transcript.append(f"{who}: {event.text.strip()}")
                del self.transcript[:-RECENT_TRANSCRIPT_LINES * 2]
            return
        if isinstance(event, rt.DelegationRequested):
            asyncio.create_task(self._delegate(event))
            return
        if isinstance(event, (rt.ProviderFailure, rt.SessionTerminated)):
            self.ended_reason = self.ended_reason or f"provider ended the session ({type(event).__name__})"
            _log.warning("live-direct: %r", event)

    async def _delegate(self, event: Any) -> None:
        prompt = (event.prompt or "").strip()
        if not prompt:
            # GPT-Live sometimes omits the prompt; fall back to the operator's last final words.
            users = [line[6:] for line in self.transcript if line.startswith("User: ")]
            prompt = users[-1] if users else ""
        if not prompt:
            await self.send([rt.SubmitDelegationResult(
                event.delegation_id,
                "No request was captured. Ask the user what they need.",
                kind="context",
            )])
            return
        self.delegations += 1
        context = VOICE_TURN_NOTE
        recent = self.recent_context()
        if recent:
            context += f"\n[Recent spoken conversation, newest last:\n{recent}]"
        turn = _PendingTurn(event.delegation_id, prompt, _norm(prompt)[:400], context)
        _LEDGER.add(turn)
        _log.info("live-direct: delegation %s -> Hermes turn (%d chars)", event.delegation_id,
                  len(prompt))
        try:
            await self.runner._handle_voice_channel_input(
                self.guild_id, self.operator_id, prompt, adapter=self.adapter
            )
        except Exception as exc:  # noqa: BLE001 - the operator must hear a failure, not silence
            _log.warning("live-direct: dispatch failed (%s)", type(exc).__name__)
            turn.done.set()
            await self.send([rt.SubmitDelegationResult(
                event.delegation_id,
                "The backend could not start that request. Tell the user it failed and to try again.",
                kind="context",
            )])
            return
        done = await asyncio.to_thread(turn.done.wait, DELEGATION_SPOKEN_TIMEOUT_S)
        if done:
            return  # _on_result already spoke it
        turn.spoken_timeout = True
        await self.send([rt.SubmitDelegationResult(
            event.delegation_id,
            "This is still running. Tell the user it is taking a while and that the full answer "
            "will be posted in the Discord thread, and you will say it when it lands.",
            kind="context",
        )])

    def on_result(self, turn: _PendingTurn) -> None:
        asyncio.create_task(self._speak_result(turn))

    async def _speak_result(self, turn: _PendingTurn) -> None:
        if self.ended_reason:
            return
        text = (turn.result or "").strip() or "The backend finished but returned no answer."
        text = text[:12000]
        self.transcript.append(f"Backend: {text[:600]}")
        if not turn.spoken_timeout:
            await self.send([rt.SubmitDelegationResult(turn.delegation_id, text, kind="context")])
        else:
            await self.send([rt.AppendLiveContext(
                f"A backend result just arrived for the earlier request \"{turn.prompt[:200]}\". "
                f"Tell the user briefly: {text}",
                kind="message",
            )])

    def status(self) -> str:
        minutes = (time.monotonic() - self.started_at) / 60
        return (f"GPT-Live (live-direct) is up in guild {self.guild_id}: {minutes:.1f} min, "
                f"{self.delegations} handoff(s) to Hermes.")


# ---------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------

def start(invoker_user_id: Any, guild_id: int | None = None) -> str:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return "GPT-Live voice needs the gateway's event loop."
    try:
        operator = int(str(invoker_user_id))
    except (TypeError, ValueError):
        return "GPT-Live voice needs an identified Discord operator."
    operators = operator_ids()
    if not operators:
        return "Set TALK_DISCORD_OPERATOR_USER_IDS before starting GPT-Live voice."
    if operator not in operators:
        return "Only a configured voice operator can start GPT-Live voice."
    if session_active() or talk_discord.native_session_active():
        return "A voice session is already running. `/talk leave` first."
    with suppress(Exception):
        existing = talk_discord._SESSION.get("task")  # legacy realtime lane
        if existing is not None and not existing.done():
            return "A Realtime voice session is already running. `/talk leave` first."
    try:
        bridge = talk_discord.resolve_voice_bridge(guild_id)
        runner = _gateway_runner()
    except talk_discord.TalkDiscordError as exc:
        return f"Can't start GPT-Live: {exc}."
    if not (getattr(bridge["adapter"], "_voice_text_channels", None) or {}).get(bridge["guild_id"]):
        return "Can't start GPT-Live: no text channel is bound to this voice call. Run `/voice join` again."

    live = LiveDirectSession(bridge=bridge, operator_id=operator, runner=runner)

    async def _run() -> None:
        failure = None
        try:
            await live.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface one sentence, log the type
            failure = str(exc) if isinstance(exc, talk_discord.TalkDiscordError) else type(exc).__name__
            _log.warning("live-direct: session failed (%s)", type(exc).__name__, exc_info=True)
        finally:
            await live.close()
            with _STATE_LOCK:
                if _STATE.get("live") is live:
                    _STATE.clear()
        if failure:
            await live.post_text(f"⚠️ GPT-Live voice stopped: {failure}")
        elif live.ended_reason is None:
            await live.post_text("📴 GPT-Live voice ended.")

    with _STATE_LOCK:
        task = loop.create_task(_run())
        _STATE.update({"task": task, "live": live, "loop": loop, "on_result": live.on_result})
    return "Starting GPT-Live voice on your Codex subscription. Give it a few seconds, then talk."


def _voice_event(runner: Any):
    """A minimal MessageEvent for the host's /voice join|leave handlers, built from the live source."""

    from gateway.platforms.event import MessageEvent
    from gateway.plugin_messages import bound_route

    source = bound_route()
    if source is None:
        raise talk_discord.TalkDiscordError("this command has no Discord source to join from")
    try:
        from gateway.session_context import get_session_env
        guild = get_session_env("HERMES_SESSION_SCOPE_ID") or ""
    except ImportError:
        guild = ""
    guild = getattr(source, "scope_id", None) or guild
    if not str(guild).isdigit():
        raise talk_discord.TalkDiscordError("voice only works inside a server, not in DMs")
    from types import SimpleNamespace

    return MessageEvent(
        text="/voice join", source=source, user_id=getattr(source, "user_id", None),
        raw_message=SimpleNamespace(guild_id=int(guild)),
    ), int(guild)


def _bot_in_voice(guild_id: int) -> bool:
    with suppress(Exception):
        talk_discord.resolve_voice_bridge(guild_id)
        return True
    return False


async def _join_and_start(invoker_user_id: Any) -> str:
    try:
        operator = int(str(invoker_user_id))
    except (TypeError, ValueError):
        return "GPT-Live voice needs an identified Discord operator."
    if operator not in operator_ids():
        return "Only a configured voice operator can start GPT-Live voice."
    joined_note = ""
    try:
        runner = _gateway_runner()
        event, guild_id = _voice_event(runner)
    except talk_discord.TalkDiscordError as exc:
        return f"Can't start GPT-Live: {exc}."
    if not _bot_in_voice(guild_id):
        joined_note = str(await runner._handle_voice_channel_join(event) or "").strip()
        if not _bot_in_voice(guild_id):
            return joined_note or "Couldn't join your voice channel. Are you in one?"
        await asyncio.sleep(0.5)  # let the receiver settle before the bridge taps it
    reply = start(operator, guild_id)
    return f"{joined_note}\n{reply}".strip() if joined_note else reply


def join_and_start(invoker_user_id: Any):
    """`/talk join`: joins the operator's voice channel if needed, then starts GPT-Live."""

    return _join_and_start(invoker_user_id)


async def _stop_and_leave() -> str:
    with _STATE_LOCK:
        task = _STATE.get("task")
        live = _STATE.get("live")
    reply = stop()
    if task is not None:
        with suppress(BaseException):
            await asyncio.wait_for(asyncio.shield(task), 5)
    with suppress(Exception):
        runner = _gateway_runner()
        event, _ = _voice_event(runner)
        await runner._handle_voice_channel_leave(event)
        return "GPT-Live voice hung up and left the voice channel."
    if live is not None:
        with suppress(Exception):
            await live.adapter.leave_voice_channel(live.guild_id)
            return "GPT-Live voice hung up and left the voice channel."
    return reply


def stop_and_leave():
    """`/talk leave`: hang up GPT-Live and leave the voice channel."""

    return _stop_and_leave()


def stop() -> str:
    with _STATE_LOCK:
        task = _STATE.get("task")
    if task is None or task.done():
        return "GPT-Live voice isn't running."
    task.cancel()
    return "GPT-Live voice hung up."


def status() -> str:
    with _STATE_LOCK:
        live = _STATE.get("live")
        task = _STATE.get("task")
    if live is None or task is None or task.done():
        return "GPT-Live voice isn't running."
    return live.status()


def reset_for_tests() -> None:
    with _STATE_LOCK:
        _STATE.clear()
    _LEDGER.clear()


__all__ = [
    "LIVE_PERSONA", "admit_packet", "audience_ids", "operator_ids", "post_llm_call_hook",
    "pre_llm_call_hook", "room_violation", "session_active", "start", "status", "stop",
]
