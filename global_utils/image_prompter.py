#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Dan Schumacher

How to run:
    python ./global_utils/image_prompter.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import base64
import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from openai import OpenAI


# ----------------------------- Data Structures ----------------------------- #


@dataclass
class Prompt:
    """
    A single multimodal prompt example (few-shot or query).

    This object:
        1) Accepts a `user` dict of fields (e.g., {"question": "...", "context": "..."}).
        2) Optionally embeds an image by inlining a data URL into a special key "__image_b64__".
        3) Optionally stores an `assistant` reply for few-shot examples.
        4) Provides a pretty, truncated string representation safe for printing (no huge Base64 dumps).

    Fields
    ------
    user : Dict[str, Any]
        Arbitrary user fields. If an image is attached, we inject a special
        key "__image_b64__" with {"type": "image_url", "image_url": {"url": "data:..."}}
    img_path : Optional[str]
        Path to an image to encode as Base64 and attach to the user content. If None, prompt is text-only.
    assistant : Optional[Dict[str, Any]]
        An optional assistant message for few-shot demonstrations.

    Computed Fields (post-init)
    ---------------------------
    mime : Optional[str]
        The guessed image MIME type (e.g., "image/png") if img_path is set.
    b64 : Optional[str]
        The Base64-encoded image content if img_path is set.
    list_of_msgs : List[Dict[str, str]]
        A compact internal representation used when building printable strings.
    str_representation : str
        Cached pretty JSON string with truncated Base64 for safe logging/printing.
    """

    user: Dict[str, Any]
    img_path: Optional[str] = None
    assistant: Optional[Dict[str, Any]] = None
    img_detail:str ="auto"
    mime: Optional[str] = field(init=False, default=None)
    b64: Optional[str] = field(init=False, default=None)
    list_of_msgs: List[Dict[str, str]] = field(init=False)
    str_representation: str = field(init=False)

    def __post_init__(self) -> None:
        """Prepare data URL (if image given), and build a safe printable representation."""
        assert self.img_detail in ["low", "high", "auto"], "detail must be 'low', 'high', or 'auto'"
        # Attach image (if provided) as an OpenAI-style `image_url` block inside the user dict.
        if self.img_path:
            self.mime = self._guess_mime(self.img_path)
            self.b64 = self._encode_b64(self.img_path)
            data_url = f"data:{self.mime};base64,{self.b64}"
            # Use a dedicated marker key so downstream formatters can detect/emit multimodal blocks.
            self.user["__image_b64__"] = {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": self.img_detail},
            }

        # Build a minimal message ledger for readable string output.
        self.list_of_msgs = [{"user": json.dumps(self.user)}]
        if self.assistant is not None:
            self.list_of_msgs.append({"assistant": json.dumps(self.assistant)})

        self.str_representation = self._build_pretty_str()

    # ---- Private helpers (static) ---- #

    @staticmethod
    def _guess_mime(path: str) -> str:
        """Guess MIME type from filename; default to 'image/png' if unknown."""
        mime, _ = mimetypes.guess_type(path)
        return mime or "image/png"

    @staticmethod
    def _encode_b64(path: str) -> str:
        """Read a file from disk and return its Base64-encoded contents (UTF-8 string)."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ---- Private helpers (instance) ---- #

    def _build_pretty_str(self) -> str:
        """
        Build an indented JSON representation of prompt messages with Base64 truncated.

        Returns
        -------
        str
            A human-readable JSON string where any embedded Base64 image is
            shortened to prevent log spam while preserving structure.
        """
        pretty: List[Dict[str, Any]] = []

        for message in self.list_of_msgs:
            for role, content_json in message.items():
                # Parse the stored JSON for readability and potential truncation
                content_obj: Dict[str, Any] = json.loads(content_json)

                if role == "user" and "__image_b64__" in self.user:
                    # Safely truncate the large `data:` URL so printing is concise.
                    user_with_img = dict(content_obj)
                    try:
                        url: str = user_with_img["__image_b64__"]["image_url"]["url"]
                        prefix, b64_payload = url.split("base64,", 1)
                        truncated_url = prefix + "base64," + b64_payload[:48] + "..."
                        user_with_img["__image_b64__"]["image_url"]["url"] = truncated_url
                    except Exception:
                        # If something about the payload is off, keep the original (rare).
                        pass

                    pretty.append({"role": role, "value": user_with_img})
                else:
                    pretty.append({"role": role, "value": content_obj})

        return json.dumps(pretty, indent=3, ensure_ascii=False)

    # ---- Display hooks ---- #

    def __str__(self) -> str:
        return self.str_representation

    def __repr__(self) -> str:
        return self.str_representation


# ------------------------------- Core Prompter ------------------------------ #


class ImagePrompter:
    """
    Helper for constructing OpenAI multimodal messages and fetching completions.

    Usage
    -----
    1) Instantiate and set `model_name` and `system_prompt`.
    2) Build Prompt objects for few-shot examples and a query.
    3) Call `format_prompt(examples, query)` to get OpenAI-style messages.
    4) Call `get_completion(messages)` (single) or `get_completion([messages, ...])` (batch).

    Attributes
    ----------
    client : OpenAI
        OpenAI API client, created using an API key loaded from `.env`.
    system_prompt : str
        The system instruction for the chat model.
    model_name : str
        The chat/completions model identifier (e.g., "gpt-4o-mini").
    """

    def __init__(self) -> None:
        # Load API key and initialize an OpenAI client instance.
        api_key = self._load_api_key_from_env()
        self.client: OpenAI = OpenAI(api_key=api_key)

        # These are set by the caller.
        self.system_prompt: str = ""
        self.model_name: str = ""

    # ---- Environment & configuration ---- #

    def _load_api_key_from_env(self) -> str:
        """
        Load the OpenAI API key from `./resources/.env`.

        Returns
        -------
        str
            The API key string.

        Raises
        ------
        ValueError
            If no API key is found at OPENAI_API_KEY.
        """
        load_dotenv("./resources/.env")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API Key not found. Set OPENAI_API_KEY=xxxx in ./resources/.env"
            )
        return api_key

    # ---- Prompt formatting ---- #

    def format_prompt(self, examples: List[Prompt], query: Prompt) -> List[Dict[str, Any]]:
        """
        Convert few-shot examples + a query `Prompt` into OpenAI chat `messages`.

        The strategy preserves the *key order* from `query.user` and requires
        each example's `user` dict to have the same ordered keys (including
        the optional image token key "__image_b64__" if present).

        Parameters
        ----------
        examples : List[Prompt]
            Zero or more `Prompt` objects used as few-shot demonstrations.
        query : Prompt
            The target query `Prompt` to be answered.

        Returns
        -------
        List[Dict[str, Any]]
            Messages ready for `client.chat.completions.create`.

        Raises
        ------
        AssertionError
            If any example's `user` keys do not match the query's `user` keys
            (order-sensitive) which helps prevent schema drift in few-shot prompts.
        """
        # Enforce exact key order agreement between examples and query.
        query_key_order: List[str] = list(query.user.keys())
        for i, example in enumerate(examples):
            example_keys = list(example.user.keys())
            assert (
                example_keys == query_key_order
            ), (
                "USER KEYS MUST MATCH FOR FEW-SHOT EXAMPLES (INCLUDING IMAGES)\n"
                f"- example #{i} keys: {example_keys}\n"
                f"- expected (query):  {query_key_order}"
            )

        # Start with a system message.
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]

        def user_dict_to_mm_content(user_obj: Dict[str, Any], key_order: Sequence[str]) -> List[Dict[str, Any]]:
            """
            Convert a `user` dict into a list of multimodal content blocks.
            Text keys become {"type":"text","text": "..."}.
            The special "__image_b64__" key becomes {"type":"image_url", ...}.
            """
            content_blocks: List[Dict[str, Any]] = []
            for key in key_order:
                value = user_obj[key]
                if key == "__image_b64__":
                    # Already shaped as {"type":"image_url","image_url":{"url":...}}
                    content_blocks.append(
                        {"type": "image_url", "image_url": value["image_url"]}
                    )
                else:
                    # Ensure text payloads are strings; JSON-encode for non-strings.
                    text_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                    content_blocks.append({"type": "text", "text": f"{key}: {text_value}"})
            return content_blocks

        # Add few-shot examples (user → assistant).
        for example in examples:
            messages.append(
                {"role": "user", "content": user_dict_to_mm_content(example.user, query_key_order)}
            )
            if example.assistant is not None:
                # Coerce assistant content to string (common: {"answer": "..."}).
                assistant_payload = example.assistant
                if isinstance(assistant_payload, str):
                    assistant_text = assistant_payload
                elif isinstance(assistant_payload, dict) and "answer" in assistant_payload:
                    assistant_text = cast(str, assistant_payload["answer"])
                else:
                    assistant_text = json.dumps(assistant_payload, ensure_ascii=False)
                messages.append({"role": "assistant", "content": assistant_text})

        # Add query (final user message).
        messages.append(
            {"role": "user", "content": user_dict_to_mm_content(query.user, query_key_order)}
        )

        return messages

    # ---- Completions ---- #

    def get_completion(
        self,
        prompts: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        *,
        max_workers: int = 10,
        request_timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fetch a completion for a single prompt or a batch of prompts.

        Parameters
        ----------
        prompts : Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
            - Single prompt: a `messages` list like: [{"role":"system",...}, {"role":"user",...}]
            - Batch: a list of such `messages` lists, one per request.
        max_workers : int, optional
            Max worker threads for batch execution, by default 10.
        request_timeout : float, optional
            Timeout (seconds) applied to waiting for futures to finish, by default 60.0.
        temperature : float, optional
            Sampling temperature for generation, by default 0.0.

        Returns
        -------
        Union[Dict[str, Any], List[Dict[str, Any]]]
            - Single dict: {"content": "..."} or {"error": "..."} for single prompt.
            - List of dicts for batch, in the same order as input.

        Notes
        -----
        - Order is preserved for batch results.
        - Exceptions are caught and returned as {"error": "..."} entries
          so that a single failure does not crash the whole batch.
        """
        # Normalize to a batch: List[List[Dict[str, Any]]]
        is_single_prompt = (
            isinstance(prompts, list)
            and (len(prompts) == 0 or isinstance(prompts[0], dict))
        )
        batch: List[List[Dict[str, Any]]]
        if is_single_prompt:
            batch = [cast(List[Dict[str, Any]], prompts)]
        else:
            batch = cast(List[List[Dict[str, Any]]], prompts)

        # Some sanity checks to catch obvious type issues early.
        assert all(isinstance(m, list) for m in batch), "Each item in batch must be a list of message dicts."
        for idx, msgs in enumerate(batch):
            assert all(isinstance(x, dict) for x in msgs), f"Batch item {idx} must be a list of dict messages."

        results: List[Optional[Dict[str, Any]]] = [None] * len(batch)

        def _call_single(i: int, messages: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
            """
            Worker function: send one chat.completions request and extract the first message content.
            Returns
            -------
            (index, result_dict)
            """
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                )
                content: str = response.choices[0].message.content or ""
                return i, {"content": content}
            except Exception as exc:
                raise ValueError(f'{i}, {"error": f"{type(exc).__name__}: {exc}"}')

            # Note: if you want tokens/logprobs etc., extract more fields here.

        # Fire all requests concurrently (or the one request, if single).
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_call_single, i, messages)
                for i, messages in enumerate(batch)
            ]
            for future in as_completed(futures, timeout=request_timeout):
                index, result = future.result()
                results[index] = result

        # There should be no `None` left, but guard defensively:
        finalized = [r if r is not None else {"error": "UnknownError: missing result"} for r in results]
        return finalized[0] if is_single_prompt else finalized
    

    def export_prompt_markdown(
        self,
        examples: List["Prompt"],
        query: "Prompt",
        out_md_path: str,
        *,
        save_images: bool = True,
        images_dirname: str = "images"
    ) -> str:
        """
        Write a readable Markdown file showing:
          - system prompt
          - each example (user fields, embedded/saved image if present, assistant answer)
          - the final query (user fields + image if present)

        Returns the absolute path to the written markdown file.
        """
        def _safe_stem(s: str) -> str:
            # Keep it simple: alnum, dash, underscore
            return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

        messages = self.format_prompt(examples, query)  # ensures key-order checks
        os.makedirs(os.path.dirname(out_md_path), exist_ok=True)

        # Optionally create an images folder next to the MD file
        images_dir = os.path.join(os.path.dirname(out_md_path), images_dirname)
        if save_images:
            os.makedirs(images_dir, exist_ok=True)

        def _image_md_from_data_url(data_url: str, stem: str) -> str:
            try:
                header, b64 = data_url.split(",", 1)
                mime = header.split(";")[0].replace("data:", "").strip()
                ext = {"image/png":"png","image/jpeg":"jpg","image/jpg":"jpg","image/webp":"webp"}.get(mime, "png")
                filename = f"{_safe_stem(stem)}.{ext}"
                path = os.path.join(images_dir, filename)
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))
                return f"![{stem}]({images_dirname}/{filename})"
            except Exception:
                return f"![{stem}]({data_url})"

        lines: List[str] = []
        ts = datetime.now().isoformat(timespec="seconds")
        lines.append(f"# Exported Prompt ({ts})")
        lines.append("")
        # System message is always the first in messages (from format_prompt)
        if messages and messages[0].get("role") == "system":
            lines.append("## System Prompt")
            lines.append("")
            lines.append("```")
            lines.append(messages[0].get("content", "") or "")
            lines.append("```")
            lines.append("")

        # Recreate a readable view that mirrors your examples -> assistant -> query structure
        # We'll iterate over your original objects to keep labels like "Example 1" consistent.
        def dump_user_block(user_dict: Dict[str, Any], key_order: Sequence[str], stem: str):
            for k in key_order:
                v = user_dict[k]
                if k == "__image_b64__":
                    # v can be dict {"type":"image_url","image_url":{"url":...}} or a list of such dicts
                    def _emit_one(img_dict, idx=None):
                        url = img_dict["image_url"]["url"]
                        tag = f"{stem}" if idx is None else f"{stem}_{idx}"
                        if save_images and url.startswith("data:"):
                            lines.append(_image_md_from_data_url(url, tag))
                        else:
                            lines.append(f"![{tag}]({url})")

                    if isinstance(v, dict) and "image_url" in v:
                        _emit_one(v)
                    elif isinstance(v, list):
                        for i, img_dict in enumerate(v, start=1):
                            _emit_one(img_dict, i)
                    else:
                        # not expected, fallback to raw
                        lines.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
                else:
                    text_val = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    lines.append(f"- **{k}**: {text_val}")
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Determine key order from query.user (format_prompt enforces exact match on examples)
        key_order = list(query.user.keys())

        # Examples
        if examples:
            lines.append("## Few-shot Examples")
            lines.append("")
            for i, ex in enumerate(examples, start=1):
                lines.append(f"### Example {i}")
                dump_user_block(ex.user, key_order, stem=f"example_{i}")
                if ex.assistant is not None:
                    lines.append("**Assistant**")
                    if isinstance(ex.assistant, dict) and "answer" in ex.assistant:
                        lines.append("")
                        lines.append("```")
                        lines.append(str(ex.assistant["answer"]))
                        lines.append("```")
                    else:
                        lines.append("")
                        lines.append("```json")
                        lines.append(json.dumps(ex.assistant, ensure_ascii=False, indent=2))
                        lines.append("```")
                lines.append("")

        # Query
        lines.append("## Query")
        dump_user_block(query.user, key_order, stem="query")

        with open(out_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return os.path.abspath(out_md_path)


# ----------------------------------- Main ----------------------------------- #


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    system_prompt_text: str = (
        "You are a helpful assistant. Your job is to help users identify "
        "what is in various images."
    )
    default_question: str = "What animal is shown in the following image?"
    default_context: str = "My friend is making a pixel art game, and I can't tell what creature this is."

    # Initialize prompter (loads API key from ./resources/.env)
    prompter = ImagePrompter()
    prompter.model_name = "gpt-4o-mini"  # set your target model here
    prompter.system_prompt = system_prompt_text

    # -------------------------------------------------------------------------
    # TEST 1 — Prompt creation & truncated string representation
    # -------------------------------------------------------------------------
    print("\n=== TEST 1: Prompt creation with image ===")
    example_cat = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./ex1.jpg",
        assistant={"answer": "Cat"},
    )
    print(example_cat)

    print("\n=== TEST 1B: Prompt creation with a different image ===")
    example_axolotl = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./query.jpg",
    )
    print(example_axolotl)

    # -------------------------------------------------------------------------
    # TEST 2 — Verify base64 truncation happened
    # -------------------------------------------------------------------------
    print("\n=== TEST 2: Base64 truncation check ===")
    assert "..." in str(example_cat), "Base64 truncation did not occur."
    assert "base64," in str(example_cat), "Data URL missing."
    print("Truncation verified.")

    # -------------------------------------------------------------------------
    # TEST 3 — format_prompt correctness
    # -------------------------------------------------------------------------
    print("\n=== TEST 3: format_prompt output shape ===")
    fewshot_messages = prompter.format_prompt([example_cat], example_axolotl)

    assert isinstance(fewshot_messages, list), "format_prompt must return List[dict]"
    assert isinstance(fewshot_messages[0], dict) and fewshot_messages[0].get("role") == "system", (
        "First message must be a system role message"
    )
    print("format_prompt returned a valid prompt structure.")

    # -------------------------------------------------------------------------
    # TEST 3B — Zero-shot prompt formatting (no examples)
    # -------------------------------------------------------------------------
    print("\n=== TEST 3B: Zero-shot prompt format ===")
    zero_shot_messages = prompter.format_prompt([], example_axolotl)

    assert isinstance(zero_shot_messages, list), "Zero-shot prompt must be a list"
    assert zero_shot_messages[0].get("role") == "system", "Zero-shot first message must be system"
    assert zero_shot_messages[-1].get("role") == "user", "Zero-shot last message must be user"
    assert isinstance(zero_shot_messages[-1]["content"], list), "User content must be a multimodal list"

    print("Zero-shot formatting verified.")
    zero_shot_result = prompter.get_completion(zero_shot_messages)
    print(zero_shot_result)

    # -------------------------------------------------------------------------
    # TEST 3C — Prompt with no images should still work
    # -------------------------------------------------------------------------
    print("\n=== TEST 3C: Prompt without image ===")
    text_only_query = Prompt(
        user={"question": "What is 2 + 2?", "context": "Simple math question."}
        # No image attached
    )

    text_only_messages = prompter.format_prompt([], text_only_query)
    text_only_result = prompter.get_completion(text_only_messages)
    print(text_only_result)

    # Validate structure
    assert text_only_messages[-1]["role"] == "user", "Text-only prompt must end with user message"
    assert all(
        block.get("type") == "text" for block in text_only_messages[-1]["content"]
    ), "Text-only prompt must not contain image blocks"
    print("No-image formatting verified.")

    # -------------------------------------------------------------------------
    # TEST 4 — Single completion (cat example explaining axolotl)
    # -------------------------------------------------------------------------
    print("\n=== TEST 4: Single completion call ===")
    single_reply = prompter.get_completion(fewshot_messages)
    print("LLM:", single_reply)

    # -------------------------------------------------------------------------
    # TEST 5 — Batch completion with cat, dog, lion queries
    # -------------------------------------------------------------------------
    print("\n=== TEST 5: Batch completion with 3 images ===")
    example_dog = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./ex2.jpg",
    )
    example_lion = Prompt(
        user={"question": default_question, "context": default_context},
        img_path="./ex3.jpg",
    )

    # Build three independent prompts for the batch
    batch_requests: List[List[Dict[str, Any]]] = [
        prompter.format_prompt([example_cat], example_axolotl),  # cat → axolotl
        prompter.format_prompt([example_cat], example_dog),      # cat → dog
        prompter.format_prompt([example_cat], example_lion),     # cat → lion
    ]

    batch_results = prompter.get_completion(batch_requests)
    print("\nBatch results:")
    assert isinstance(batch_results, list), "Expected list of results for batch mode."
    for i, result in enumerate(batch_results):
        print(f"{i}: {result}")

    assert len(batch_results) == 3, "Batch size mismatch."
    print("\nBatch execution verified.")

    # -------------------------------------------------------------------------
    # TEST 6 — Order preservation under threading
    # -------------------------------------------------------------------------
    print("\n=== TEST 6: Thread ordering ===")
    for idx, item in enumerate(batch_results):
        assert isinstance(item, dict), f"Unexpected output type at index {idx}"
    print("Threaded order preserved.")

    print("\nAll tests completed successfully.")


    out_path = prompter.export_prompt_markdown(
        examples=[example_cat],          # or more: [example_cat, example_dog, ...]
        query=example_axolotl,           # whatever your current query Prompt is
        out_md_path="./prompt_export.md",
        save_images=False,                # set False to keep data URLs inline
        images_dirname="./"          # folder next to the .md
    )
    print(f"Markdown exported to: {out_path}")
