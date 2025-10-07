# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0

"""
PDF Reader Function for NAT/AIQ
Reads PDF files and extracts text content
"""

import logging
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PDFReaderConfig(FunctionBaseConfig, name="pdf_reader"):
    """Configuration for PDF reader function"""
    description: str = Field(
        default="Reads a PDF file and extracts text content",
        description="Description of the PDF reader function"
    )
    max_pages: int | None = Field(
        default=None,
        description="Maximum number of pages to read (None for all pages)"
    )


@register_function(config_type=PDFReaderConfig)
async def pdf_reader_function(config: PDFReaderConfig, builder: Builder):
    """
    Create a PDF reader function that extracts text from PDF files.
    
    Args:
        config: PDFReaderConfig with function settings
        builder: Builder instance for accessing other components
        
    Yields:
        FunctionInfo for the PDF reader
    """
    
    async def read_pdf(pdf_path: str) -> str:
        """
        Read and extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file (can be local or URL)
            
        Returns:
            Extracted text content from the PDF
        """
        try:
            # Import PyMuPDF (fitz) for PDF processing
            import fitz  # PyMuPDF
            
            logger.info(f"Reading PDF from: {pdf_path}")
            
            # Check if it's a local file
            path = Path(pdf_path)
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract text from each page
            text_content = []
            max_pages = config.max_pages or len(doc)
            
            for page_num in range(min(len(doc), max_pages)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}")
            
            doc.close()
            
            # Combine all text
            full_text = "\n\n".join(text_content)
            
            if not full_text.strip():
                return "No text content found in PDF"
            
            logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except ImportError:
            error_msg = ("PyMuPDF not installed. Please install with: "
                        "pip install PyMuPDF")
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error reading PDF: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    # Return the function info
    yield FunctionInfo.from_fn(
        read_pdf,
        description=config.description
    )