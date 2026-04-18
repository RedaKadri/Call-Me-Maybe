import json

import numpy as np

from llm_sdk import Small_LLM_Model

from src.schemas import FunctionDefinition


class ConstrainedDecoder:
    def __init__(self, functions_definitions: list[FunctionDefinition]) -> None:
        self.model = Small_LLM_Model()

        self.functions_definitions = [
            {
                "name": fn.name,
                "description": fn.description,
                "parameters": {
                    key: value.model_dump() for key, value in fn.parameters.items()
                },
                "returns": fn.returns.model_dump(),
            }
            for fn in functions_definitions
        ]

        self.valid_function_names = [
            fn["name"] for fn in self.functions_definitions
        ] + ["none"]

        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.loads(f.read())

        self.stop_tokens_id = {151643, 151645}

        self.state = "EXPECT_FUNCTION_NAME"

    def _apply_constraints(self, logits: np.array, output: str) -> np.array:
        if self.state == "EXPECT_FUNCTION_NAME":
            ...

        return logits

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        output = '{"name": "'
        print(output, end="")

        input_ids = self.model.encode(prompt + output)[0].tolist()

        for _ in range(max_tokens):
            logits = np.array(self.model.get_logits_from_input_ids(input_ids))

            logits = self._apply_constraints(logits, output)
            next_token_id = int(np.argmax(logits))

            if next_token_id in self.stop_tokens_id:
                break

            input_ids.append(next_token_id)
            token_text = self.model.decode([next_token_id])
            output += token_text

            print(token_text, end="", flush=True)
        print()

        return output
