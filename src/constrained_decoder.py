import json

import numpy as np

from llm_sdk import Small_LLM_Model

from src.schemas import FunctionDefinition


class ConstrainedDecoder:
    def __init__(
        self, functions_definitions: list[FunctionDefinition]
    ) -> None:
        self.model = Small_LLM_Model()

        self.functions_definitions = [
            {
                "name": fn.name,
                "description": fn.description,
                "parameters": {
                    key: value.model_dump()
                    for key, value in fn.parameters.items()
                },
                "returns": fn.returns.model_dump(),
            }
            for fn in functions_definitions
        ]
        self.valid_functions_names = [
            *(str(fn["name"]) for fn in self.functions_definitions), "none"
        ]

        with open(self.model.get_path_to_vocab_file()) as f:
            self.vocab = json.loads(f.read())

        self.prompt_prefix = '{"name": "'
        self.stop_tokens_id = {151643, 151645}

        self.state: str = "EXPECT_FUNCTION_NAME"

    def _is_valid_function_name(self, name: str) -> bool:
        for valid_function_name in self.valid_functions_names:
            if valid_function_name.startswith(name):
                return True
        return False

    def _get_allowed_tokens_id(self, curr: str) -> list[int]:
        if self.state == "EXPECT_FUNCTION_NAME":
            return [
                id
                for token, id in self.vocab.items()
                if self._is_valid_function_name(curr + token)
            ]

        return []

    def _apply_constraints(
        self, logits: np.ndarray, output: str
    ) -> np.ndarray:
        constrained_logits = logits.copy()

        match self.state:
            case "EXPECT_FUNCTION_NAME":
                curr = output[len(self.prompt_prefix):]
                allowed_ids = self._get_allowed_tokens_id(curr)

            case "EXPECT_FUNCTION_NAME_END":
                allowed_ids = [self.vocab['"}'], self.vocab['",']]

            case _:
                allowed_ids = []

        if allowed_ids:
            constrained_logits = np.full_like(logits, -np.inf)
            constrained_logits[allowed_ids] = logits[allowed_ids]

        return constrained_logits

    def _update_state(self, output: str) -> None:
        if self.state == "EXPECT_FUNCTION_NAME":
            curr = output[len(self.prompt_prefix):]
            if curr in self.valid_functions_names:
                self.state = "EXPECT_FUNCTION_NAME_END"

        elif self.state == "EXPECT_FUNCTION_NAME_END":
            if output.endswith('",') or output.endswith('"}'):
                self.state = "EXPECT_PARAMETERS_START"

    def generate(self, prompt: str, max_tokens: int = 64) -> str:
        self.state = "EXPECT_FUNCTION_NAME"

        output = self.prompt_prefix
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

            self._update_state(output)

            print(token_text, end="", flush=True)

        print()

        return output
