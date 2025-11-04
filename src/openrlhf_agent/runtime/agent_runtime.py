import json
from typing import Dict, List, Optional

from openrlhf_agent.runtime.engine import LLMEngine
from openrlhf_agent.environment import Environment
from openrlhf_agent.template import Template
from openrlhf_agent.runtime.session import AgentSession


class AgentRuntime:
    def __init__(
        self,
        engine: LLMEngine,
        environment: Environment,
        template: Template,
        *,
        max_new_tokens_per_step: int = 10240,
    ):
        self.engine = engine
        self.session = AgentSession(environment, template)
        self.environment = self.session.environment
        self.template = self.session.template
        self.max_new_tokens_per_step = max_new_tokens_per_step

    @staticmethod
    def _is_internal_obs(text: str) -> bool:
        """
        Internal observations are JSON payloads with __internal=true (by our env design).
        Tool outputs (think/final) are plain text; they should NOT be JSON.
        """
        try:
            data = json.loads(text)
            return isinstance(data, dict) and (data.get("__internal") is True or data.get("visible_to_user") is False)
        except Exception:
            return False

    def run_steps(self, messages: List[Dict[str, str]]):
        """
        Streaming generator. Yields dicts for UI/logging:
          - {"role": "tool_call", "name": ..., "arguments": {...}}
          - {"role": "tool_response", "content": "..."}      # visible only
          - {"role": "assistant", "content": "..."}          # final answer
          - {"role": "error", "content": "..."}              # parse errors, timeouts, etc.
        """
        self.session.reset()

        prompt = self.template.render_messages(
            [{"role": "system", "content": self.environment.system_prompt}, *messages],
            tools_manifest=self.environment.tools_manifest(),
            add_generation_prompt=True,
        )
        prompt_ids = self.engine.tokenize(prompt)

        for _ in range(self.environment.max_steps):
            action_ids, action_text = self.engine.generate(
                prompt_ids, max_tokens=self.max_new_tokens_per_step
            )

            step_result = self.session.step_from_text(action_text, runtime=True)

            if step_result.parse_error:
                yield {"role": "_inner", "content": action_text}
                observation_ids = self.engine.tokenize(step_result.rendered_observation)
                prompt_ids += action_ids + observation_ids
                continue

            for action in step_result.actions or []:
                if action is None:
                    continue
                yield {"role": "tool_call", "name": action.name, "arguments": action.arguments}

            for obs in step_result.observations:
                if not self._is_internal_obs(obs):
                    yield {"role": "tool_response", "content": obs}

            observation_ids = self.engine.tokenize(step_result.rendered_observation)
            prompt_ids += action_ids + observation_ids

            if step_result.terminated:
                final_text = ""
                for o in reversed(step_result.observations):
                    if not self._is_internal_obs(o):
                        final_text = o
                        break
                yield {"role": "assistant", "content": final_text}
                return

        yield {"role": "error", "content": "Max steps reached without final."}

    def run_final(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Convenience wrapper: returns the final assistant content or None.
        """
        for step in self.run_steps(messages):
            if step.get("role") == "assistant":
                return step.get("content")
        return None
