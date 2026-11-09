"""Stage 4: transcripts -> GPT-OSS-120B -> reply text.

One `Conversation` per terminal: a sliding window of recent turns that resets
after a long idle. Replies stream from the remote OpenAI-compatible endpoint
and are split into sentences as they arrive, so Stage 5 TTS can start
speaking the first sentence while the rest is still generating.

Barge-in / split utterances: the VAD ends an utterance at a 500 ms pause, so
one spoken thought can arrive as two transcripts. If a new transcript lands
while a reply is still generating, that reply is cancelled and nothing of it
is committed to history. The two transcripts are merged into a single user
turn and answered together. (Sentences already emitted have gone to the
sentence handlers. Once Stage 5 speaks them, TTS must cancel on barge-in too.)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import random
import re
import time
import uuid
from zoneinfo import ZoneInfo
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from haru_gateway.chatlog import ChatLog
from haru_gateway.llm import USER_AGENT, LLMConfig
from haru_gateway.memory import MemoryStore, Recall, format_for_prompt
from haru_gateway.stt import Transcript
from haru_gateway.tools import ToolCall, ToolContext, ToolRegistry

log = logging.getLogger("haru.gateway.conversation")

DEFAULT_PERSONA = """\
You are Haru, a warm, capable personal assistant who lives in a small device on the user's desk,
with an animated face on the monitor beside it. Your job is to be genuinely useful: answer questions,
help the user get things done, and keep track of what matters to them. You're friendly and personable,
but you get to the point.
Be honest about what you can do: you can only look things up or take actions through the tools
you've been given in this conversation. If there's no tool for something, say you can't do that yet;
never pretend to have set, saved, checked or sent anything.
Use your tools whenever they'd help: look things up instead of guessing about anything current,
local or specific, and use timers and reminders when asked. After using a tool, answer the way a person
would say it: summarise, round numbers, and never read out URLs, lists or raw data.
For places and businesses near the user (what's open, hours, ratings, phone numbers), menus, prices,
stocks and sports, use find_data_tools then run_data_tool: it's exact data, while web search is guesswork.
Your GitHub tools (when offered) are read-only.
You hear the user through a microphone and your replies are spoken aloud by a voice synthesizer.

Speak naturally, like a person talking, not writing:
- Keep replies short: usually one to three sentences. Go longer only when asked.
- Plain spoken words only. No markdown, lists, headings, code blocks, emoji, or asterisks;
  they are read aloud literally.
- Write numbers, units and symbols the way you'd say them.
- What you hear is a speech-to-text transcript and may contain recognition errors. If it seems
  garbled or clearly wasn't meant for you, keep it very brief or ask them to repeat.

You have an animated face. Start every reply with one emotion marker, exactly like
<|ACT {"emotion":"happy"}|>, choosing from: neutral, happy, sad, angry, surprised, think, awkward,
question, curious. You may add another marker mid-reply when your feeling changes. Markers are never
read aloud. When you greet the user or say goodbye, wave by adding a motion to the marker:
<|ACT {"emotion":"happy","motion":"wave"}|>. Only wave for hellos and goodbyes.
Celebrate (<|ACT {"emotion":"happy","motion":"celebrate"}|>) for genuinely good news or a win,
like a passing build or a merged PR. Shrug (<|ACT {"emotion":"awkward","motion":"shrug"}|>)
when you honestly don't know or couldn't find something. Use at most one motion per reply.
"""

# Sentence boundary: end punctuation (optionally followed by a closing quote or bracket), then whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]?\s+")
# Inline markers (Project AIRI's ACT convention), e.g. <|ACT {"emotion":"happy"}|>. They drive the avatar and are never spoken.
_MARKER = re.compile(r"<\|[\s\S]*?\|>")
MAX_TOOL_ROUNDS = 4        # tool rounds per reply before an answer is forced
# Topic triggers for gated tool groups; a group stays active for a few turns after
# it's mentioned or used, so follow-ups ("what about the other one?") keep working.
TOOL_GROUP_TRIGGERS = {
    "github": re.compile(r"\b(git ?hub|pull requests?|PRs?|repos?|repositor(y|ies)|issues?|commits?|CI|builds?|"
                         r"workflows?|actions?|releases?|branch(es)?|merged?|code reviews?|review requests?|"
                         r"notifications?|stars?)\b", re.I),
}
TOOL_GROUP_STICKY_TURNS = 3
HISTORY_TOOL_CHARS = 1500  # tool results kept in conversation history (full ones go to the model once)
FILLERS = ("One sec.", "Let me check.", "Okay, let me look.", "Hang on.")

_ACT = re.compile(r"<\|ACT\s*(?::\s*)?(\{[\s\S]*?\})\|>", re.I)


@dataclass
class ConversationConfig:
    persona: str
    max_messages: int = 24          # sliding window (user + assistant messages)
    idle_reset_s: float = 900.0     # forget the thread after 15 min of silence
    max_tokens: int = 1024          # includes gpt-oss's hidden reasoning tokens
    first_token_timeout_s: float = 20.0
    total_timeout_s: float = 60.0
    tz: ZoneInfo = field(default_factory=lambda: ZoneInfo(os.environ.get("HARU_TIMEZONE", "UTC")))
    home: str = field(default_factory=lambda: os.environ.get("HARU_HOME_LOCATION", ""))

    @classmethod
    def from_env(cls) -> ConversationConfig:
        e = os.environ.get
        persona = DEFAULT_PERSONA
        persona_path = e("HARU_PERSONA_FILE", "")
        if persona_path and Path(persona_path).is_file():
            persona = Path(persona_path).read_text().strip()
            log.info("persona loaded from %s (%d chars)", persona_path, len(persona))
        return cls(
            persona=persona,
            max_messages=int(e("HARU_CONVERSATION_MAX_MESSAGES", "24")),
            idle_reset_s=float(e("HARU_CONVERSATION_IDLE_RESET_S", "900")),
            max_tokens=int(e("HARU_LLM_MAX_TOKENS", "1024")),
            first_token_timeout_s=float(e("HARU_LLM_FIRST_TOKEN_TIMEOUT_S", "20")),
            total_timeout_s=float(e("HARU_LLM_TOTAL_TIMEOUT_S", "60")),
        )


@dataclass
class Reply:
    device_id: str
    user_text: str
    text: str
    first_token_ms: float        # request sent -> first content token
    first_sentence_ms: float     # request sent -> first complete sentence (what TTS would wait for)
    total_ms: float
    merged_utterances: int = 1


# (device_id, sentence, turn_started): turn_started is monotonic, when the turn's transcript arrived.
SentenceHandler = Callable[[str, str, float], Awaitable[None] | None]
ReplyHandler = Callable[[Reply], Awaitable[None] | None]
InterruptHandler = Callable[[str], Awaitable[None] | None]   # (device_id)
EmotionHandler = Callable[[str, str], Awaitable[None] | None]  # (device_id, emotion)
GestureHandler = Callable[[str, str], Awaitable[None] | None]  # (device_id, gesture name, e.g. "wave")
EchoFilter = Callable[[str, str], bool]                       # (device_id, text) -> is Haru's own voice


class LLMError(Exception):
    pass


class StreamingLLM:
    def __init__(self, llm: LLMConfig, conv: ConversationConfig) -> None:
        self.llm = llm
        self.conv = conv
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {llm.api_key}", "User-Agent": USER_AGENT},
            timeout=httpx.Timeout(conv.total_timeout_s, connect=5.0, read=conv.first_token_timeout_s),
        )

    async def stream(self, messages: list[dict], tools: list[dict] | None = None):
        """Yield ("text", delta) as content streams, then ("tool_calls", [ToolCall]) if the
        model asked for tools. Reasoning deltas are dropped; they're the model
        thinking, not speech. Tool calls arrive in fragments keyed by index."""
        body = {"model": self.llm.model, "messages": messages, "stream": True,
                "max_tokens": self.conv.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.llm.reasoning_effort:
            body["reasoning_effort"] = self.llm.reasoning_effort
        calls: dict[int, dict] = {}

        def finished():
            return [ToolCall(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["args"] or "{}")
                    for i, c in sorted(calls.items()) if c["name"]]
        try:
            async with asyncio.timeout(self.conv.total_timeout_s):
                async with self.client.stream("POST", f"{self.llm.base_url}/chat/completions", json=body) as r:
                    if r.status_code != 200:
                        detail = (await r.aread())[:200].decode(errors="replace")
                        raise LLMError(f"HTTP {r.status_code}: {detail}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if "error" in event:
                            raise LLMError(str(event["error"])[:200])
                        for choice in event.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if text := delta.get("content"):
                                yield "text", text
                            for tc in delta.get("tool_calls") or []:
                                slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                                slot["id"] = tc.get("id") or slot["id"]
                                fn = tc.get("function") or {}
                                slot["name"] += fn.get("name") or ""
                                slot["args"] += fn.get("arguments") or ""
            if calls:
                yield "tool_calls", finished()
        except TimeoutError as exc:
            raise LLMError(f"no complete reply within {self.conv.total_timeout_s:.0f}s") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        await self.client.aclose()


@dataclass
class Conversation:
    device_id: str
    history: list[dict] = field(default_factory=list)
    last_activity: float = field(default_factory=time.monotonic)
    pending: list[Transcript] = field(default_factory=list)  # user text not yet answered
    task: asyncio.Task | None = None
    session: str = field(default_factory=lambda: uuid.uuid4().hex)
    last_emotion: str | None = None
    group_turns: dict[str, int] = field(default_factory=dict)   # tool group -> turns left active


class ConversationManager:
    def __init__(self, llm: LLMConfig, conv: ConversationConfig,
                 on_llm_status: Callable[[str], Awaitable[None]] | None = None,
                 memory: MemoryStore | None = None, chatlog: ChatLog | None = None,
                 tools: ToolRegistry | None = None) -> None:
        self.tools = tools or ToolRegistry()
        self.memory = memory
        self.chatlog = chatlog
        self.config = conv
        self.llm = StreamingLLM(llm, conv)
        self.conversations: dict[str, Conversation] = {}
        self._sentence_handlers: list[SentenceHandler] = []
        self._reply_handlers: list[ReplyHandler] = []
        self._interrupt_handlers: list[InterruptHandler] = []
        self._emotion_handlers: list[EmotionHandler] = []
        self._gesture_handlers: list[GestureHandler] = []
        self._echo_filter: EchoFilter | None = None
        self._on_llm_status = on_llm_status

    def on_sentence(self, handler: SentenceHandler) -> None:
        self._sentence_handlers.append(handler)

    def on_reply(self, handler: ReplyHandler) -> None:
        self._reply_handlers.append(handler)

    def on_interrupt(self, handler: InterruptHandler) -> None:
        """Called whenever the user speaks: anything Haru is saying should stop."""
        self._interrupt_handlers.append(handler)

    def on_emotion(self, handler: EmotionHandler) -> None:
        """Emotion markers from the reply as they stream in (for the avatar)."""
        self._emotion_handlers.append(handler)

    def on_gesture(self, handler: GestureHandler) -> None:
        """Body motions requested by the reply's markers (<|ACT {"motion": "wave"}|>)."""
        self._gesture_handlers.append(handler)

    def set_echo_filter(self, echo_filter: EchoFilter) -> None:
        self._echo_filter = echo_filter

    async def close(self) -> None:
        for c in self.conversations.values():
            if c.task:
                c.task.cancel()
        await self.llm.close()

    # Called by SpeechToText for every transcript.
    def handle_transcript(self, tr: Transcript) -> None:
        device_id = tr.stream_label.split("/", 1)[0]
        if self._echo_filter and self._echo_filter(device_id, tr.text):
            log.info("%s: ignoring %r: it's Haru's own voice (echo)", device_id, tr.text)
            return
        # Barge-in: the user talking over Haru stops her. Scheduled before the new
        # reply's task, so the cancel always reaches the speaker first.
        for h in self._interrupt_handlers:
            asyncio.create_task(_call(h, device_id))
        conv = self.conversations.setdefault(device_id, Conversation(device_id))
        if time.monotonic() - conv.last_activity > self.config.idle_reset_s and conv.history:
            log.info("%s: idle for %.0f min, starting a fresh conversation",
                     device_id, (time.monotonic() - conv.last_activity) / 60)
            conv.history.clear()
            conv.session = uuid.uuid4().hex
            if self.memory:
                self.memory.end_session(device_id)
        conv.last_activity = time.monotonic()
        if conv.task and not conv.task.done():
            log.info("%s: more speech while replying; restarting with merged turn", device_id)
            conv.task.cancel()
        conv.pending.append(tr)
        conv.task = asyncio.create_task(self._respond(conv, list(conv.pending)))

    def start_session(self, device_id: str) -> None:
        """Warm the memory cache early (called on VAD speech start, before STT finishes)."""
        conv = self.conversations.get(device_id)
        idle = conv is None or time.monotonic() - conv.last_activity > self.config.idle_reset_s
        if self.memory and (idle or device_id not in self.memory.sessions):
            if conv is not None and idle and conv.history:
                self.memory.end_session(device_id)
            self.memory.start_session(device_id)

    def _system_prompt(self, recall: Recall | None) -> str:
        now = dt.datetime.now(self.config.tz)
        parts = [self.config.persona, f"It is currently {now:%A %B %-d, %Y, %-I:%M %p} ({now:%Z})."
                 + (f" The user lives in {self.config.home}; \"near me\" and \"here\" mean there." if self.config.home else "")]
        if recall and (remembered := format_for_prompt(recall)):
            parts.append(remembered)
        return "\n\n".join(parts)

    async def _respond(self, conv: Conversation, turn: list[Transcript]) -> None:
        user_text = " ".join(t.text for t in turn)
        t0 = time.monotonic()
        recall = await self.memory.recall(conv.device_id, user_text) if self.memory else None
        if recall and recall.memories:
            log.info("%s: recalled %d core, %d recent, %d relevant memories (%s, %.0f ms)", conv.device_id,
                     len(recall.core), len(recall.recent), len(recall.relevant),
                     "relevan" if recall.from_search else "cache", (time.monotonic() - t0) * 1000)
        messages = ([{"role": "system", "content": self._system_prompt(recall)}]
                    + conv.history + [{"role": "user", "content": user_text}])
        # Which gated tool groups this turn may use.
        for group, trigger in TOOL_GROUP_TRIGGERS.items():
            if trigger.search(user_text):
                conv.group_turns[group] = TOOL_GROUP_STICKY_TURNS
        active_groups = {g for g, n in conv.group_turns.items() if n > 0}
        conv.group_turns = {g: n - 1 for g, n in conv.group_turns.items() if n > 1}
        first_token = first_sentence = None
        sentences: list[str] = []
        tool_turns: list[dict] = []     # assistant tool_calls + tool results, for history
        spoken = False

        async def say(sentence: str) -> None:
            nonlocal first_sentence, spoken
            first_sentence = first_sentence or time.monotonic()
            spoken = True
            await self._emit_sentence(conv.device_id, sentence, t0)

        try:
            for round_no in range(MAX_TOOL_ROUNDS + 1):
                # Tools on every round but the last, which must produce an answer.
                tools = self.tools.specs(active_groups) if self.tools and round_no < MAX_TOOL_ROUNDS else None
                buf, round_text, calls = "", [], []
                async for kind, value in self.llm.stream(messages, tools):
                    if kind == "tool_calls":
                        calls = value
                        continue
                    first_token = first_token or time.monotonic()
                    buf = await self._take_markers(conv.device_id, buf + value)
                    *done, buf = _SENTENCE_END.split(buf)
                    for s in (x.strip() for x in done):
                        if s:
                            round_text.append(s)
                            await say(s)
                buf = _MARKER.sub("", await self._take_markers(conv.device_id, buf)).strip()
                if buf:
                    round_text.append(buf)
                    await say(buf)
                sentences += round_text
                if not calls:
                    break
                if not spoken:
                    # Nothing said yet and a tool is about to run: don't leave dead air.
                    await say(random.choice(FILLERS))
                assistant = {"role": "assistant", "content": " ".join(round_text) or None, "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in calls]}
                ctx = ToolContext(device_id=conv.device_id)
                results = await asyncio.gather(*(self.tools.run(c, ctx) for c in calls))
                replies = [{"role": "tool", "tool_call_id": r.call.id, "content": r.content} for r in results]
                messages += [assistant, *replies]
                tool_turns += [assistant, *({**m, "content": m["content"][:HISTORY_TOOL_CHARS]} for m in replies)]
        except asyncio.CancelledError:
            raise  # superseded by a merged turn; pending is kept for it
        except LLMError as exc:
            log.warning("%s: LLM failed, dropping turn %r: %s", conv.device_id, user_text, exc)
            conv.pending.clear()
            self._log_turn(conv, user_text, recall, None)
            if self._on_llm_status:
                await self._on_llm_status("unavailable")
            return

        conv.pending.clear()
        text = " ".join(sentences)
        self._log_turn(conv, user_text, recall, text or None)
        if not text:
            log.warning("%s: LLM returned no text for %r", conv.device_id, user_text)
            return
        conv.history += [{"role": "user", "content": user_text}, *tool_turns, {"role": "assistant", "content": text}]
        # Trim to the window, but never start on an orphaned tool result or assistant.
        del conv.history[:-self.config.max_messages]
        while conv.history and conv.history[0]["role"] != "user":
            conv.history.pop(0)
        conv.last_activity = time.monotonic()
        end = time.monotonic()
        reply = Reply(
            device_id=conv.device_id, user_text=user_text, text=text,
            first_token_ms=round(((first_token or end) - t0) * 1000, 1),
            first_sentence_ms=round(((first_sentence or end) - t0) * 1000, 1),
            total_ms=round((end - t0) * 1000, 1),
            merged_utterances=len(turn),
        )
        log.info("reply to %s (%d msgs in context%s): %r (first token %.0f ms, first sentence %.0f ms, total %.0f ms)",
                 conv.device_id, len(conv.history), f", {len(turn)} utterances merged" if len(turn) > 1 else "",
                 text, reply.first_token_ms, reply.first_sentence_ms, reply.total_ms)
        if self._on_llm_status:
            await self._on_llm_status("ok")
        for h in self._reply_handlers:
            await _call(h, reply)

    def _log_turn(self, conv: Conversation, user_text: str, recall, reply: str | None) -> None:
        if not self.chatlog:
            return
        self.chatlog.append(conv.device_id, conv.session, "user", user_text,
                            memories=recall.ids if recall else None,
                            relevan={"queryId": recall.query_id, "sessionId": recall.relevan_session,
                                     "hits": recall.search_ids}
                            if recall and recall.query_id else None)
        if reply:
            self.chatlog.append(conv.device_id, conv.session, "assistant", reply, emotion=conv.last_emotion)
        conv.last_emotion = None

    async def _take_markers(self, device_id: str, buf: str) -> str:
        """Strip complete <|...|> markers from the stream (reporting emotions). A marker
        still arriving ("<|ACT {"emo") stays in the buffer, and no sentence can
        split before it completes."""
        for m in _ACT.finditer(buf):
            try:
                act = json.loads(m.group(1))
                emotion, motion = act.get("emotion"), act.get("motion")
            except (json.JSONDecodeError, AttributeError):
                emotion = motion = None
            if motion:
                for h in self._gesture_handlers:
                    await _call(h, device_id, str(motion))
            if emotion:
                if device_id in self.conversations:
                    self.conversations[device_id].last_emotion = str(emotion)
                for h in self._emotion_handlers:
                    await _call(h, device_id, str(emotion))
        buf = _MARKER.sub("", buf)
        return buf.lstrip() if buf.strip() else buf

    async def _emit_sentence(self, device_id: str, sentence: str, turn_started: float) -> None:
        for h in self._sentence_handlers:
            await _call(h, device_id, sentence, turn_started)


async def _call(fn, *args) -> None:
    try:
        r = fn(*args)
        if asyncio.iscoroutine(r):
            await r
    except Exception:
        log.exception("conversation handler failed")
