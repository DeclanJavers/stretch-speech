"""The LLM: Qwen 3.5 4B running in LM Studio.

Every reply is forced to JSON with a fixed schema:
    {"name": str | null, "item": "pen" | "shirt" | "plushie" | "none" | null, "say": str}

  item = one of the items  -> person asked for it, conversation can end
  item = "none"            -> person declined, conversation can end
  item = null              -> still chatting / unclear, keep talking

Two LM Studio details that matter:
  * reasoning_effort="none" turns Qwen's thinking off. (chat_template_kwargs
    enable_thinking=False is ignored by LM Studio.) With thinking on, a reply takes ~10 s.
  * response_format=json_schema makes LM Studio constrain decoding, so the
    output always parses and item is always a legal value.
"""

import json
import time

from openai import OpenAI

import config

# Kept short on purpose: reading the prompt is most of the LLM's delay
# (~540 tokens took 1.4 s before the first word; ~200 tokens takes ~0.5 s).
SYSTEM_PROMPT = """You are Stretch, a cheerful robot giving away free swag at a convention, \
one item per visitor. In stock: {item_list}.{out_of_stock} Your words are spoken aloud.

Reply in JSON:
- say: one or two short sentences, under 25 words, no emojis.
- name: the first name the visitor said, else null. Never guess a name.
- item: what the visitor decided, matching what you just said:
  {item_keys} = they clearly chose exactly one available item;
  "declined" = they turned you down ("no thanks", "I'm good", "nope", "I'll pass", "bye");
  "undecided" = anything else: a question, chat, unsure, asked for several or something you don't have.

You are Stretch; the visitor is someone else, so never use their name as yours. A question \
about an item ("do you have ...?", "what do you have?") is not a choice: answer and ask which they want. \
If you're guessing what they meant, use "undecided" and ask "Did you say ...?". When they chose, \
confirm it and say you'll bring it right back. You don't know the venue, so for directions say \
to ask a staff member. Their words come from speech recognition and may have errors: \
"tea shirt" is t-shirt, "plush e" is plushie, "stuffed guy" is plushie.

Examples:
Visitor: "Hi, I'm Alex. What do you have?"
{{"say": "Nice to meet you, Alex! I have {spoken_items}. Which one would you like?", "name": "Alex", "item": "undecided"}}
Visitor: "Can I get the {example_item}?"
{{"say": "{example_spoken}, great choice! I'll bring it right back.", "name": null, "item": "{example_item}"}}"""


class Brain:
    def __init__(self, model: str = config.LLM_MODEL, base_url: str = config.LLM_BASE_URL):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=config.LLM_TIMEOUT_S)

    def check(self) -> None:
        """Fail fast with a helpful message if LM Studio isn't serving the model."""
        try:
            ids = [m.id for m in self.client.models.list().data]
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"Can't reach the model server at {self.client.base_url} ({e}).\n"
                "LM Studio:  lms server start   then   lms load <model> -c 2048\n"
                "llama.cpp:  llama-server -m <file.gguf> --alias <model> -c 2048 --jinja --port 8081"
            )
        if self.model not in ids:
            raise SystemExit(f"Model '{self.model}' isn't available. Loaded/known: {ids}\n"
                             f"Run:  lms load {self.model}")
        # The first request after loading is slow (~11 s); pay that now, not on the first visitor.
        self.respond([{"role": "user", "content": "Hi!"}])

    @staticmethod
    def assistant_turn(say: str, name: str | None, item: str | None) -> dict:
        """A past robot reply, in the same JSON shape the model is asked to produce."""
        decision = {"none": "declined", None: "undecided"}.get(item, item)
        return {"role": "assistant", "content": json.dumps({"say": say, "name": name, "item": decision})}

    @staticmethod
    def system_prompt(available_items: list[str]) -> str:
        item_list = ", ".join(f'{config.ITEM_DESCRIPTIONS[i]} ("{i}")' for i in available_items)
        gone = [config.ITEM_DESCRIPTIONS[i].split(" (")[0] for i in config.ALL_ITEMS if i not in available_items]
        out = f" You ran out of {' and '.join(gone)}." if gone else ""
        keys = " / ".join(f'"{i}"' for i in available_items)
        spoken = [config.ITEM_DESCRIPTIONS[i].split(" (")[0] for i in available_items]  # "a t-shirt"
        spoken_items = spoken[0] if len(spoken) == 1 else ", ".join(spoken[:-1]) + ", and " + spoken[-1]
        example = available_items[-1]  # examples only ever mention items that are in stock
        return SYSTEM_PROMPT.format(item_list=item_list, out_of_stock=out, item_keys=keys,
                                    spoken_items=spoken_items, example_item=example,
                                    example_spoken=spoken[-1][0].upper() + spoken[-1][1:])

    @staticmethod
    def schema(available_items: list[str]) -> dict:
        return {
            "type": "object",
            # "say" comes first: the model writes its reply, then labels it. Small models
            # label far more consistently that way than when they pick the item first.
            "properties": {
                "say": {"type": "string"},
                "name": {"type": ["string", "null"]},
                # "declined"/"undecided" read far less ambiguously to a small model than
                # "none"/null did (decline accuracy went from 1/9 with those).
                "item": {"enum": [*available_items, "declined", "undecided"]},
            },
            "required": ["say", "name", "item"],
            "additionalProperties": False,
        }

    def respond(self, history: list[dict], available_items: list[str] | None = None) -> dict:
        """history: [{"role": "assistant"|"user", "content": str}, ...] (no system message).

        Returns {"name", "item", "say", "llm_s", "error"}. Never raises.
        """
        items = available_items or config.ALL_ITEMS
        # Qwen's chat template errors ("No user query found") if the conversation
        # starts with an assistant turn, and Stretch always speaks first.
        opener = [{"role": "user", "content": "(A visitor walks up to you.)"}] \
            if history and history[0]["role"] == "assistant" else []
        messages = [{"role": "system", "content": self.system_prompt(items)}, *opener, *history]
        start = time.perf_counter()
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "strict": True, "schema": self.schema(items)},
                },
                # LM Studio turns thinking off via reasoning_effort; llama.cpp's server via
                # chat_template_kwargs. Each ignores the other, so send both.
                extra_body={"reasoning_effort": "none",
                            "chat_template_kwargs": {"enable_thinking": False}},
            )
            msg = r.choices[0].message
            # Thinking models in LM Studio can put the JSON in reasoning_content instead.
            raw = msg.content or (msg.model_extra or {}).get("reasoning_content") or ""
            out = json.loads(raw)
            decision = out.get("item")
            item = {"declined": "none", "undecided": None}.get(decision, decision)
            if item not in [*items, "none", None]:
                item = None
            name = (out.get("name") or "").strip() or None
            say = (out.get("say") or "").strip() or "Sorry, could you say that again?"
            return {"name": name, "item": item, "say": say,
                    "llm_s": round(time.perf_counter() - start, 2), "error": None}
        except Exception as e:  # noqa: BLE001 - a robot should keep talking, not crash
            return {"name": None, "item": None, "say": "Sorry, I missed that. Could you say it again?",
                    "llm_s": round(time.perf_counter() - start, 2), "error": repr(e)}
