#!/usr/bin/env python3
"""
Prompt template system for Guac sequence-to-sequence training.

This module defines special tokens and Jinja templates for structured model inputs
with proper sequence boundaries, guidance markers, and inference delimiters.
"""

from jinja2 import Template, Environment, FileSystemLoader
from typing import Dict, List, Set, Any, Optional
import os

# Flag tokens (single tokens for state/mode changes)
FLAG_TOKEN_NAMES = {
    "PAD",          # Padding token
    "UNK",          # Unknown token
    "RESET",        # Session reset marker

    # Reasoning toggle
    "THINK", "NOTHINK",

    # Out-of-band broadcast
    "PING", "ALERT", "PANIC",

    # Diagnostic
    "AFFIRMATIVE", "NEGATIVE"
}

# Serial tokens (paired start/end tokens for content boundaries)
SERIAL_TOKEN_NAMES = {
    # Guidance property boundaries
    "GUIDE_SYS", "GUIDE_ORG", "GUIDE_USR",

    # Task boundaries
    "TASK",

    # Sequence boundaries and property containers
    "SEQUENCE",
    "LABEL",
    "MODAL_HINT",

    # Inference markers
    "INFERENCE",
}

# Template properties (replaced with actual content)
TEMPLATE_PROPERTIES = {
    "system_guidance", "organization_guidance", "user_guidance",
    "task",
    "sequence.label",
    "sequence.modality_hint",
}

class PromptTokenizer:
    """Manages special tokens and template rendering for Guac prompts."""

    def __init__(self, template_dir=None):
        self.flag_tokens = {}
        self.serial_tokens = {}
        self.template_props = {}

        # Set up template directory
        if template_dir is None:
            # Default to templates directory relative to this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            template_dir = os.path.join(project_root, "templates")

        self.template_dir = template_dir
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))

        # Generate flag tokens
        for name in FLAG_TOKEN_NAMES:
            self.flag_tokens[name] = f"<& {name} &>"

        # Generate serial tokens (start/end pairs)
        for name in SERIAL_TOKEN_NAMES:
            self.serial_tokens[f"{name}_start"] = f"<& {name}/ &>"
            self.serial_tokens[f"{name}_end"] = f"<& /{name} &>"

        # Generate template properties
        for prop in TEMPLATE_PROPERTIES:
            self.template_props[prop] = "{{ " + prop + " }}"

    def get_all_special_tokens(self) -> List[str]:
        """Return all special tokens for tokenizer vocabulary."""
        tokens = []
        tokens.extend(self.flag_tokens.values())
        tokens.extend(self.serial_tokens.values())
        return tokens

    def get_core_tokens(self) -> Dict[str, str]:
        """Return core tokens needed by tokenizer (PAD, UNK, etc.)."""
        return {
            'pad_token': self.flag_tokens['PAD'],
            'unk_token': self.flag_tokens['UNK'],
            'bos_token': self.flag_tokens['RESET'],
            'eos_token': self.serial_tokens['INFERENCE_end'],
        }

    def render_template(self, template_string: str, context: Dict[str, Any]) -> str:
        """Render a Jinja template with prompt tokens and context."""
        template = Template(template_string)
        render_context = {
            "f": self.flag_tokens,      # Flag tokens
            "s": self.serial_tokens,    # Serial tokens
            "p": self.template_props,   # Property placeholders
            **context                   # User-provided context
        }
        return template.render(render_context)

    def render_template_file(self, template_filename: str, context: Dict[str, Any]) -> str:
        """Render a Jinja template file with prompt tokens and context."""
        template = self.jinja_env.get_template(template_filename)
        render_context = {
            "f": self.flag_tokens,      # Flag tokens
            "s": self.serial_tokens,    # Serial tokens
            "p": self.template_props,   # Property placeholders
            **context                   # User-provided context
        }
        return template.render(render_context)

    def list_available_templates(self) -> List[str]:
        """List all available template files."""
        if not os.path.exists(self.template_dir):
            return []
        return [f for f in os.listdir(self.template_dir) if f.endswith('.jinja')]

# Default template configuration
DEFAULT_TEMPLATE_FILE = "default.jinja"

# Template rules (for documentation)
TEMPLATE_RULES = """\
1.  Every session shall begin with a reset token
2.  An inference begin token shall be appended to every session immediately preceding inference
3.  An inference end token shall be expected to signal an intended cessation of inference
"""

def create_prompt_tokenizer() -> PromptTokenizer:
    """Create a new prompt tokenizer instance."""
    return PromptTokenizer()

def apply_default_template(context: Dict[str, Any]) -> str:
    """Apply the default template file with given context."""
    tokenizer = create_prompt_tokenizer()
    return tokenizer.render_template_file(DEFAULT_TEMPLATE_FILE, context)

def apply_template_file(template_filename: str, context: Dict[str, Any]) -> str:
    """Apply a specific template file with given context."""
    tokenizer = create_prompt_tokenizer()
    return tokenizer.render_template_file(template_filename, context)

# For backward compatibility and quick testing
if __name__ == "__main__":
    tokenizer = create_prompt_tokenizer()

    # Example context
    example_context = {
        #"system_guidance": "",
        #"organization_guidance": "",
        #"user_guidance": "",
        #"task": "",
        #"sequence.label": "",
        #"sequence.modality_hint": "text"
        "sequences": [
            {

            },
            {
                
            }
        ]
    }

    rendered = apply_default_template(example_context)

    print("Rendered template:")
    print(rendered)
    print("---")
    print(f"{len(tokenizer.get_all_special_tokens())} special tokens defined")
    print("---")
    print("Special tokens:", tokenizer.get_all_special_tokens()[:10], "...")