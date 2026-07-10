#!/usr/bin/env python3
"""
Script to render the default.jinja template with example data using PromptTokenizer.
Demonstrates the multi-turn sequence-to-sequence template functionality.
"""

import json
import sys
import os

# Add the scripts/training directory to the path to import prompt_templates
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts', 'training'))

from prompt_templates import create_prompt_tokenizer, apply_default_template

def main():
    # Create the prompt tokenizer which provides proper special tokens
    tokenizer = create_prompt_tokenizer()

    # Example context data for the template
    example_context = {
        # Guidance content
        "system_guidance": "",
        "organization_guidance": "",
        "user_guidance": "",

        # Task definition
        "task": "Answer the following question based on the conversation history.",

        # Multi-turn conversation data
        # Note: The template structure suggests sequences is a global list that gets repeated for each turn
        "turns": [
            {
                "source": "user",
                "sequences": [
                    {
                        "label": "user_input",
                        "modality_hint": "text",
                        "content": "What is the capital of France?"
                    },
                ]
            },
            {
                "source": "inference",
                "sequences": [
                    {
                        "label": "assistant_response",
                        "modality_hint": "text",
                        "content": "The capital of France is Paris."
                    },
                ]
            },
            {
                "source": "user",
                "sequences": [
                    {
                        "label": "user_input",
                        "modality_hint": "text",
                        "content": "What is the capital of Canada?"
                    },
                ]
            },
        ],

        # Whether to add generation prompt
        "add_generation_prompt": True
    }

    # Render the template using the proper function from prompt_templates.py
    rendered = apply_default_template(example_context)

    print("=== Rendered Template Output ===")
    print(rendered)

    print("\n" + "="*50 + "\n")
    print("=== Core Tokenizer Tokens ===")
    print("Core tokens for tokenizer:", json.dumps(tokenizer.get_core_tokens(), indent=2))

if __name__ == "__main__":
    main()