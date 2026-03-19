import sys
import json

from pydantic import ValidationError

# from pydantic_core import from_json
import numpy as np

from llm_sdk import Small_LLM_Model

from src.constrained_decoding import ConstrainedDecoding
from src.parser import config_parser, parse_args
from src.schemas import FunctionDefinition, Prompt


class IDK:
    def __init__(
        self, functions_calls: list[FunctionDefinition], prompts: list[Prompt]
    ) -> None:
        self.model = Small_LLM_Model()

        self.functions_calls = functions_calls
        self.prompts = prompts

        self._stop_ids = {151643, 151645}

        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.loads(f.read())

        self.constrained_decoding = ConstrainedDecoding(self.functions_calls)

    def _format_prompt(self, user_message: str) -> str:
        system = (
            "你是一个函数调用助手。\n"
            f"<tools>\n{self.functions_calls}\n</tools>\n"
            '如果没有合适的函数可以完成任务，只输出：{"name": "none"}\n'
            "否则，只输出："
            '{"name": "function_name", "parameters": {"key": "value"}}\n'
            "且不要输出任何其他内容：\n"
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n</think>\n\n"
        )

    def use(self) -> None:
        for prompt in self.prompts:
            res = '{"name": "'
            user_prompt = self._format_prompt(prompt)
            input_ids = self.model.encode((user_prompt + res))[0].tolist()

            while True:
                logits = np.array(
                    self.model.get_logits_from_input_ids(input_ids)
                )
                next_token_id = int(np.argmax(logits))

                if next_token_id in self._stop_ids:
                    break

                input_ids.append(next_token_id)
                token_text = self.model.decode([next_token_id])
                res += token_text

            print(res)

    def bla(self, func_call, logits):
        allowed_ids = self.model.encode('{"')

        mask = np.zeros_like(logits, dtype=bool)
        mask[allowed_ids] = True

        logits[~mask] = -np.inf


if __name__ == "__main__":
    try:
        config = parse_args(sys.argv[1:])
    except ValueError as e:
        print(
            f"Error: {e}\n"
            "Usage: uv run python -m src "
            "[--functions_definition <path>] "
            "[--input <path>] "
            "[--output <path>]"
        )
        sys.exit(1)

    try:
        idc = config_parser(config)
    except FileNotFoundError as e:
        print(f"Error: File not found — {e.filename}")
        sys.exit(1)
    except PermissionError as e:
        print(f"Error: Permission denied reading '{e.filename}'")
        sys.exit(1)
    except ValidationError as e:
        print(e)
        sys.exit(1)

    try:
        x = IDK(idc["functions_definition"], idc["prompts"])
        import time
        import datetime

        start = time.time()
        x.use()
        end = time.time() - start
        print(f"\n{datetime.timedelta(seconds=end)}")
    except (KeyboardInterrupt, EOFError):
        pass
