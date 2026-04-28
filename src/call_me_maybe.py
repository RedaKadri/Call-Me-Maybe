import json

from pydantic_core import from_json

from src.constrained_decoder import ConstrainedDecoder
from src.schemas import FunctionDefinition, Prompt


class CallMeMaybe:
    def __init__(
            self,
            functions_definitions: list[FunctionDefinition],
            prompts: list[Prompt]
    ) -> None:
        self.functions_definitions = functions_definitions
        self.prompts = prompts

        self.decoder = ConstrainedDecoder(functions_definitions)

    def _format_prompt(self, user_message: str) -> str:
        system = (
            "你可以调用一个或多个函数来协助处理用户请求。\n"
            "在 <tools></tools> 中提供了函数签名：\n"
            f"<tools>\n{self.functions_definitions}\n</tools>\n\n"
            "返回一个包含函数名称和参数的 JSON 对象：\n"
            "{\"name\": \"function_name\", \"parameters\": {\"key\": \"value\"}}\n\n"
            "如果没有可用于该请求的函数，返回：\n"
            "{\"name\": \"none\"}\n\n"
            "当参数是正则表达式时，请生成一个通用的正则模式 "
            "（例如 \\\\d+、[0-9]+、[a-z]+），而不是输入中的字面示例。"
            "该正则必须匹配输入中的所有相关项，而不仅仅是一个。"
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{json.dumps(user_message)}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n</think>\n\n"
        )

    def run(self) -> None:
        results = []

        for prompt in self.prompts:
            formatted_prompt = self._format_prompt(prompt.prompt)

            output = self.decoder.generate(formatted_prompt)

            try:
                parsed = from_json(output)
            except json.JSONDecodeError:
                parsed = {"name": "none"}

            results.append({
                "prompt": prompt.prompt,
                "name": parsed.get("name", "none"),
                "parameters": parsed.get("parameters", {})
            })

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
