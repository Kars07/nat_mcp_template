# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""
LLM Summarizer Function for NAT/AIQ
Uses an LLM to generate detailed summaries of text
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LLMSummarizerConfig(FunctionBaseConfig, name="llm_summarizer"):
    """Configuration for LLM-based summarizer function"""
    description: str = Field(
        default="Generates a detailed summary of the provided text",
        description="Description of the summarizer function"
    )
    llm_name: LLMRef = Field(
        description="Name of the LLM to use for summarization"
    )
    max_summary_length: int = Field(
        default=500,
        description="Target maximum length for the summary in words"
    )
    include_key_points: bool = Field(
        default=True,
        description="Whether to include bullet points of key insights"
    )
    summary_style: str = Field(
        default="comprehensive",
        description="Style of summary: 'brief', 'comprehensive', or 'detailed'"
    )


@register_function(config_type=LLMSummarizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def llm_summarizer_function(config: LLMSummarizerConfig, builder: Builder):
    """
    Create an LLM-based summarizer function.
    
    Args:
        config: LLMSummarizerConfig with function settings
        builder: Builder instance for accessing LLMs
        
    Yields:
        FunctionInfo for the summarizer
    """
    
    # Get the LLM from the builder
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def summarize_text(text: str) -> str:
        """
        Generate a detailed summary of the provided text using an LLM.
        
        Args:
            text: The text content to summarize
            
        Returns:
            A formatted summary of the text
        """
        try:
            # Build the prompt based on configuration
            style_instructions = {
                "brief": "Create a brief, concise summary in 2-3 sentences.",
                "comprehensive": "Create a comprehensive summary that captures all major points.",
                "detailed": "Create a detailed summary with thorough analysis of all key aspects."
            }
            
            style_instruction = style_instructions.get(
                config.summary_style, 
                style_instructions["comprehensive"]
            )
            
            key_points_instruction = ""
            if config.include_key_points:
                key_points_instruction = "\n\nAfter the summary, provide 5-7 key points as bullet points."
            
            prompt = f"""You are an expert at analyzing and summarizing documents.

Please analyze the following text and provide a summary.

{style_instruction}
Target length: approximately {config.max_summary_length} words.{key_points_instruction}

Format your response as:

## Summary
[Your summary here]
{
"## Key Points" + chr(10) + "- [Point 1]" + chr(10) + "- [Point 2]" + chr(10) + "..." if config.include_key_points else ""
}

Text to summarize:
---
{text[:10000]}  # Limit input to avoid token limits
{"..." if len(text) > 10000 else ""}
---

Summary:"""
            
            logger.info(f"Generating summary using LLM (style: {config.summary_style})")
            
            # Call the LLM
            response = await llm.ainvoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                summary = response.content
            else:
                summary = str(response)
            
            logger.info(f"Generated summary of {len(summary)} characters")
            return summary
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Return the function info
    yield FunctionInfo.from_fn(
        summarize_text,
        description=config.description
    )