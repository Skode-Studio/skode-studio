"""
LLM Utility functions for DeepCode Project

This module provides common LLM-related utitilies to avoid circular imports
and reduce code duplication across the project.
"""

from typing import Type, Any


# Import LLM classes
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
import os
import yaml


def get_preferred_llm_class(
  config_path: str = "mcp_agent.secrets.yaml"
) -> Type[Any]:
  """
  Automatically select the LLM class based on API key availability in configuration.

  Reads from YAML config file and returns AnthropicAugmentedLLM if anthropic.api_key
  is available, otherwise returns OpenAIAugmentedLLM.

  Args:
      config_path: Path to the YAML configuration file

  Returns:
      class: The preferred LLM class
  """
  
  try:
    
    # Try to read the configuration file
    if os.path.exists(config_path):
      with open(config_path, "r", "utf-8") as f:
        config = yaml.safe_load(f)
        
      
      # Check for anthropic API Key in config
      anthropic_config = config.get("anthropic", {})
      anthropic_key = anthropic_config.get("api_key", "")
      
      if anthropic_key and anthropic_key.strip() and not anthropic_key == "":
        return AnthropicAugmentedLLM
      else:
        return OpenAIAugmentedLLM
      
    else:
      print(f"🤖 Config file {config_path} not found, using OpenAIAugmentedLLM")
      return OpenAIAugmentedLLM
    
  except Exception as e:
    print(f"🤖 Error reading config file {config_path}: {e}")
    print("🤖 Falling back to OpenAIAugmentedLLM")
    return OpenAIAugmentedLLM