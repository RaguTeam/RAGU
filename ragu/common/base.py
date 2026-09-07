"""
Base class for RAGU modules that use LLM.

This class provides unified management of Jinja-based prompt templates
(:class:`PromptTemplate`) used across different components of the RAGU
system.

Each generative module declares which prompt templates it uses, either by
referencing their names from :data:`DEFAULT_PROMPT_TEMPLATES`.

Classes
-------
- :class:`RaguGenerativeModule` — Manages prompt templates for modules that
  perform LLM-driven generation or structured response tasks.
"""

from ragu.common.prompts import  DEFAULT_PROMPT_TEMPLATES
from ragu.common.prompts.prompt_storage import RAGUInstruction


class RaguGenerativeModule:
    """
    Base class for generative components in the RAGU framework.

    Provides consistent handling of prompt templates across modules.
    The class can load default templates by name or accept custom
    :class:`PromptTemplate` instances directly.
    """

    def __init__(self, prompts: list[str] | dict[str, RAGUInstruction]):
        """
        Initialize the generative module with one or more prompts.

        :param prompts: Either a list of prompt names (loaded from
                        :data:`DEFAULT_PROMPT_TEMPLATES`) or a dictionary
                        mapping prompt names to :class:`ChatTemplate` objects.
        :raises ValueError: If the input format is neither list nor dict, or if a
                            requested prompt name is not a known default prompt.
        """
        super().__init__()

        if isinstance(prompts, list):
            unknown = [name for name in prompts if name not in DEFAULT_PROMPT_TEMPLATES]
            if unknown:
                raise ValueError(
                    f"Unknown prompt name(s): {', '.join(unknown)}. "
                    f"Available prompts: {', '.join(sorted(DEFAULT_PROMPT_TEMPLATES))}"
                )
            self.prompts: dict[str, RAGUInstruction] = {
                prompt_name: DEFAULT_PROMPT_TEMPLATES[prompt_name] for prompt_name in prompts
            }
        elif isinstance(prompts, dict):
            self.prompts = prompts
        else:
            raise ValueError(
                f"Prompts must be a list of prompt names or a dictionary of prompt names and ChatTemplate objects, "
                f"got {type(prompts)}"
            )

    def get_prompts(self) -> dict:
        """
        Retrieve all prompt templates registered in the module.

        :return: Dictionary mapping prompt names to :class:`ChatTemplate` objects.
        """
        return self.prompts

    def get_prompt(self, prompt_name: str) -> RAGUInstruction:
        """
        Retrieve a specific prompt template by name.

        :param prompt_name: The name of the prompt to retrieve.
        :return: The corresponding :class:`ChatTemplate` instance.
        :raises ValueError: If the prompt name is not found.
        """
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
        else:
            raise ValueError(f"Prompt {prompt_name} not found")

    def update_prompt(self, prompt_name: str, prompt: RAGUInstruction) -> None:
        """
        Replace or add a prompt template in the module.

        :param prompt_name: The key name under which to store the prompt.
        :param prompt: The :class:`PromptTemplate` object to register.
        """
        self.prompts[prompt_name] = prompt
