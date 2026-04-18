from src.constrained_decoder import ConstrainedDecoder
from src.schemas import FunctionDefinition, Prompt


class CallMeMaybe:
    def __init__(
        self, functions_definitions: list[FunctionDefinition], prompts: list[Prompt]
    ) -> None:

        self.functions_definitions = functions_definitions
        self.prompts = prompts

        self.decoder = ConstrainedDecoder(functions_definitions)

    def _format_prompt(self, user_message: str) -> str:
        system = (
            "你是一个函数调用助手。\n"
            f"<tools>\n{self.functions_definitions}\n</tools>\n"
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

    def run(self) -> None:
        for prompt in self.prompts:
            formatted_prompt = self._format_prompt(prompt.prompt)

            output = self.decoder.generate(formatted_prompt)

            break
