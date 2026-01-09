# src/schemas/request.py
from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class GeminiModels(str, Enum):
    """
    An enumeration of the available Gemini models.
    """

    # Gemini 3.0 Series
    PRO_3_0 = "gemini-3.0-pro"

    # Gemini 2.5 Series
    PRO_2_5 = "gemini-2.5-pro"
    FLASH_2_5 = "gemini-2.5-flash"


class GeminiRequest(BaseModel):
    message: str
    model: GeminiModels = Field(default=GeminiModels.FLASH_2_5, description="Model to use for Gemini.")
    files: Optional[List[str]] = []

class OpenAIChatRequest(BaseModel):
    messages: List[dict]
    model: Optional[GeminiModels] = None
    stream: Optional[bool] = False

class Part(BaseModel):
    text: str

class Content(BaseModel):
    parts: List[Part]

class GoogleGenerativeRequest(BaseModel):
    contents: List[Content]


class SubtitleLine(BaseModel):
    """A single subtitle line with index and content."""
    index: int = Field(..., description="Line index/number")
    content: str = Field(..., description="Subtitle text content")


class MultiTranslateRequest(BaseModel):
    """
    Request for multi-line subtitle translation.
    
    Supports batch translation of subtitle lines with configurable batch size
    and parallel processing.
    """
    lines: List[SubtitleLine] = Field(..., description="List of subtitle lines to translate")
    target_language: str = Field(default="Vietnamese", description="Target language for translation")
    source_language: Optional[str] = Field(default=None, description="Source language (auto-detect if not specified)")
    model: GeminiModels = Field(default=GeminiModels.FLASH_2_5, description="Model to use for translation")
    lines_per_request: int = Field(default=10, ge=1, le=50, description="Number of lines per translation request")
    parallel_requests: int = Field(default=3, ge=1, le=10, description="Number of parallel requests to make")
    system_instruction: Optional[str] = Field(default=None, description="Custom system instruction for translation")
    prompt_template: Optional[str] = Field(
        default=None, 
        description="Custom prompt template. Use {lines} placeholder for subtitle content"
    )


class TranslatedLine(BaseModel):
    """A translated subtitle line."""
    index: int
    original: str
    translated: str


class MultiTranslateResponse(BaseModel):
    """Response containing translated subtitle lines."""
    translations: List[TranslatedLine]
    total_lines: int
    successful_lines: int
    failed_lines: int
    errors: Optional[List[str]] = None
