# Live-direct: GPT-Live on Discord without a Talk-enabled host

Upstream Hermes Talk runs GPT-Live (`gpt-live-1-codex`) on Discord only when the host includes
`gateway/discord_task_context.py`. Stock Hermes doesn't have that file. Live-direct is a plugin-only
lane for those hosts.

- **Voice:** GPT-Live on the Codex subscription (OAuth only, never an API key), carried over the
  existing `DiscordAudio` bridge.
- **Brain:** each client delegation becomes an ordinary Hermes turn in the text channel bound by
  `/voice join`, sent via the gateway's voice-input path. A `post_llm_call` hook captures the final
  reply and hands it back to GPT-Live to speak.
- **Room policy (plugin-enforced):** only `TALK_DISCORD_OPERATOR_USER_IDS` speech reaches the model.
  Everyone else's audio is replaced with silence of the same length. If anyone not on
  `DISCORD_ALLOWED_USERS` / `TALK_DISCORD_AUDIENCE_USER_IDS` joins the channel (bots excepted), the
  session ends.

## Configuration

```
TALK_VOICE_MODE=live
TALK_LIVE_AUTH=subscription
TALK_DISCORD_OPERATOR_USER_IDS=<your discord id>
TALK_LIVE_DISCORD_DIRECT=auto   # auto: only when the host lacks discord_task_context; true/false to force
TALK_LIVE_DISCORD_NAME=Hermes   # name the voice persona uses
TALK_LIVE_DISCORD_INSTRUCTIONS= # optional extra persona text
TALK_LIVE_DISCORD_TURN_TIMEOUT_S=120
TALK_LIVE_DISCORD_PREBUFFER_MS=80
```

Install the `[live]` extra (`aiortc`) into the Python environment the **gateway** actually runs in.

## Usage

Join a voice channel, then run `/voice join` followed by `/talk join`. Run `/talk leave` to hang up.

## Known limitations

- Hermes ≤ 0.22.7 folds plugin slash commands into a running agent turn as steer text, so `/talk join`
  sent while the agent is busy never reaches the plugin. This needs a host fix in
  `gateway/run_inbound.py` (`_hm_busy_slash_or_photo`) to dispatch plugin commands on the busy path.
- Audio runs at full band end to end. GPT-Live's WebRTC track is resampled once (by PyAV) straight to
  Discord's 48 kHz stereo. Mic capture stays at 48 kHz (only L/R are averaged). An 80 ms jitter
  buffer (`TALK_LIVE_DISCORD_PREBUFFER_MS`) re-arms after each underrun, so late packets do not
  click mid-word. The upstream Realtime/native lanes keep their 24 kHz contract unchanged.
- The upstream task-switching and project features of the canonical lane are not available here.
