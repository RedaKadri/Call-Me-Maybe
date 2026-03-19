# stupid code

```py
class CallMeMaybe:
    def __init__(self) -> None:
        self._model = Small_LLM_Model()

        self._stop_ids = set()
        for special in ["<|im_end|>", "<|endoftext|>"]:
            ids = self._model.encode(special)[0].tolist()
            if ids:
                self._stop_ids.add(ids[0])

    def _format_prompt(self, user_message: str) -> str:
        fn_call = {
                "name": "fn_add_numbers",
                "description": "Add two numbers together and return their sum.",
                "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
                "returns": {"type": "number"},
            }

        system = (
            "You are a helpful assistant. You are precise, factual, and concise.\n\n"
            "# Behavior Rules\n"
            "- Only state facts you are certain about.\n"
            "- If you don't know something, say: \"I don't know.\"\n"
            "- Never make up information, names, dates, or values.\n"
            "# Tool Usage\n"
            "When calling a tool, respond ONLY with this format and nothing else — no explanation, no extra text:\n"
            "<tool_call>\nname: tool or functionname\nparameters: tool or function parameters</tool_call>\n\n"
            "After receiving a tool result, use it to answer. "
            "Do NOT answer from memory if a tool can provide the information.\n\n"
            f"Available tools:{json.dumps(fn_call)}\n"
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def chat(self, user_message: str) -> str:
        prompt = self._format_prompt(user_message)
        input_ids = self._model.encode(prompt)[0].tolist()
        print(len(self._model._tokenizer))

        result = []

        while True:
            logits = self._model.get_logits_from_input_ids(input_ids)
            next_token_id = logits.index(max(logits))

            if next_token_id in self._stop_ids:
                break

            input_ids.append(next_token_id)
            token_text = self._model.decode([next_token_id])
            result.append(token_text)
            print(token_text, end="", flush=True)

        print()
        return "".join(result)
```
