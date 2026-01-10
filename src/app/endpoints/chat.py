# src/app/endpoints/chat.py
import time
import asyncio
import re
from typing import List, Tuple, Optional, TYPE_CHECKING
from fastapi import APIRouter, HTTPException
from app.logger import logger
from schemas.request import (
    GeminiRequest, 
    OpenAIChatRequest, 
    MultiTranslateRequest, 
    MultiTranslateResponse,
    TranslatedLine,
    SubtitleLine
)
from app.services.gemini_client import get_gemini_client, GeminiClientNotInitializedError
from app.services.session_manager import get_translate_session_manager

if TYPE_CHECKING:
    from models.gemini import MyGeminiClient

router = APIRouter()

@router.post("/translate")
async def translate_chat(request: GeminiRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_manager = get_translate_session_manager()
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager is not initialized.")
    try:
        # This call now correctly uses the fixed session manager
        response = await session_manager.get_response(request.model, request.message, request.files)
        return {"response": response.text}
    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")

def convert_to_openai_format(response_text: str, model: str, stream: bool = False):
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk" if stream else "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

@router.post("/v1/chat/completions")
async def chat_completions(request: OpenAIChatRequest):
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    is_stream = request.stream if request.stream is not None else False

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # Build conversation prompt with system prompt and full history
    conversation_parts = []

    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        if role == "system":
            conversation_parts.append(f"System: {content}")
        elif role == "user":
            conversation_parts.append(f"User: {content}")
        elif role == "assistant":
            conversation_parts.append(f"Assistant: {content}")

    if not conversation_parts:
        raise HTTPException(status_code=400, detail="No valid messages found.")

    # Join all parts with newlines
    final_prompt = "\n\n".join(conversation_parts)

    if request.model:
        try:
            response = await gemini_client.generate_content(message=final_prompt, model=request.model.value, files=None)
            return convert_to_openai_format(response.text, request.model.value, is_stream)
        except Exception as e:
            logger.error(f"Error in /v1/chat/completions endpoint: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing chat completion: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Model not specified in the request.")


def _build_translation_prompt(
    lines: List[SubtitleLine],
    target_language: str,
    source_language: Optional[str],
    system_instruction: Optional[str],
    prompt_template: Optional[str]
) -> str:
    """Build the translation prompt for a batch of subtitle lines."""
    
    # Format lines as "index: content"
    # Preserve explicit line breaks with escaped markers so parsing stays reliable.
    lines_text = "\n".join(
        [f"{line.index}: {line.content.replace(chr(10), r'\\n')}" for line in lines]
    )
    
    # Use custom prompt template if provided
    if prompt_template:
        return prompt_template.replace("{lines}", lines_text)
    
    # Build default prompt
    source_lang_text = f"from {source_language} " if source_language else ""
    
    system_text = ""
    if system_instruction:
        system_text = f"System instruction: {system_instruction}\n\n"
    
    prompt = f"""{system_text}You are a professional subtitle translator. Translate the following subtitle lines {source_lang_text}to {target_language}.

IMPORTANT RULES:
1. Maintain the exact same format: "index: translated content"
2. Preserve the original line numbers/indices exactly
3. Only translate the content, not the index numbers
4. Keep the translation natural and fluent for subtitles
5. Preserve any "\\n" markers inside a line as line breaks
6. Do not add any explanations or extra text
7. Return ONLY the translated lines in the same format

Input subtitles:
{lines_text}

Output (translated to {target_language}):"""
    
    return prompt


def _parse_translation_response(
    response_text: str,
    original_lines: List[SubtitleLine]
) -> List[TranslatedLine]:
    """Parse the translation response and extract translated lines.
    
    Handles multiple formats:
    1. One line per index: "1201: content"
    2. Multiple per line with pipe delimiter: "1201: content | 1202: content"
    3. Multiline content continuation
    """
    
    translated = []
    original_map = {line.index: line.content for line in original_lines}
    parsed_indices = set()
    
    # First, normalize the response by splitting on | delimiter
    # This handles format like: "1201: content | 1202: content"
    normalized_lines = []
    for raw_line in response_text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```"):
            continue
        
        # Split by | but only if followed by a digit (to avoid splitting content with |)
        # Pattern: " | " followed by digits
        segments = re.split(r'\s*\|\s*(?=\d+\s*[:\.\)\-])', line)
        for segment in segments:
            segment = segment.strip()
            if segment:
                normalized_lines.append(segment)
    
    current_idx = None
    current_content = []

    def flush_current():
        nonlocal current_idx, current_content
        if current_idx is None:
            return
        content = "\n".join(current_content).strip()
        if not content:
            return
        content = content.replace(r"\n", "\n")
        if current_idx in original_map and current_idx not in parsed_indices:
            translated.append(TranslatedLine(
                index=current_idx,
                original=original_map[current_idx],
                translated=content
            ))
            parsed_indices.add(current_idx)
        current_idx = None
        current_content = []

    for line in normalized_lines:
        # Match format: "index: content" or "index. content" or "index) content" or "index- content"
        match = re.match(r'^(\d+)\s*[:\.\)\-]\s*(.*)$', line)
        if match:
            flush_current()
            try:
                current_idx = int(match.group(1))
            except ValueError:
                current_idx = None
                current_content = []
                continue
            content = match.group(2).strip()
            current_content = [content] if content else []
            continue

        # Continuation line for current index
        if current_idx is not None:
            current_content.append(line)

    flush_current()
    
    return translated


async def _translate_batch(
    gemini_client: "MyGeminiClient",
    lines: List[SubtitleLine],
    target_language: str,
    source_language: Optional[str],
    model_value: str,
    system_instruction: Optional[str],
    prompt_template: Optional[str],
    max_retries: int = 3
) -> Tuple[List[TranslatedLine], List[str]]:
    """Translate a batch of subtitle lines with retry logic."""
    
    errors = []
    translated = []
    
    prompt = _build_translation_prompt(
        lines, target_language, source_language,
        system_instruction, prompt_template
    )
    
    last_exception = None
    for attempt in range(max_retries):
        try:
            response = await gemini_client.generate_content(
                message=prompt, 
                model=model_value, 
                files=None
            )
            
            translated = _parse_translation_response(response.text, lines)
            
            # Check if we got all translations
            if len(translated) < len(lines):
                missing = set(l.index for l in lines) - set(t.index for t in translated)
                if missing:
                    errors.append(f"Missing translations for indices: {missing}")
            
            # Success - break out of retry loop
            break
                    
        except Exception as e:
            last_exception = e
            error_type = type(e).__name__
            
            # Check if it's a network error that should be retried
            if "ReadError" in error_type or "ConnectError" in error_type or "TimeoutException" in error_type:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2, 4, 6 seconds
                    logger.warning(f"Network error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
            
            # Non-retryable error or last attempt
            error_msg = f"Translation batch failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            break
    else:
        # All retries exhausted
        if last_exception:
            error_msg = f"Translation batch failed after {max_retries} retries: {str(last_exception)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
        
    return translated, errors


@router.post("/multi-translate", response_model=MultiTranslateResponse)
async def multi_translate(request: MultiTranslateRequest):
    """
    Translate multiple subtitle lines with parallel processing.
    
    This endpoint supports:
    - Batch translation of subtitle lines
    - Configurable batch size (lines per request)
    - Parallel processing for faster translation
    - Custom system instructions and prompt templates
    
    Request format:
    ```json
    {
        "lines": [
            {"index": 1, "content": "Hello world"},
            {"index": 2, "content": "How are you?"}
        ],
        "target_language": "Vietnamese",
        "source_language": "English",  // optional
        "model": "gemini-2.5-flash",
        "lines_per_request": 10,
        "parallel_requests": 3,
        "system_instruction": "Translate naturally",  // optional
        "prompt_template": "Translate: {lines}"  // optional
    }
    ```
    
    Response format:
    ```json
    {
        "translations": [
            {"index": 1, "original": "Hello world", "translated": "Xin chào thế giới"},
            {"index": 2, "original": "How are you?", "translated": "Bạn khỏe không?"}
        ],
        "total_lines": 2,
        "successful_lines": 2,
        "failed_lines": 0,
        "errors": null
    }
    ```
    """
    try:
        gemini_client = get_gemini_client()
    except GeminiClientNotInitializedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    if not request.lines:
        raise HTTPException(status_code=400, detail="No subtitle lines provided.")
    
    # Split lines into batches
    batches = []
    for i in range(0, len(request.lines), request.lines_per_request):
        batches.append(request.lines[i:i + request.lines_per_request])
    
    logger.info(f"Multi-translate: {len(request.lines)} lines in {len(batches)} batches, "
                f"{request.parallel_requests} parallel requests")
    
    all_translations = []
    all_errors = []
    
    # Process batches with limited parallelism
    semaphore = asyncio.Semaphore(request.parallel_requests)
    
    async def process_batch(batch: List[SubtitleLine]):
        async with semaphore:
            return await _translate_batch(
                gemini_client,
                batch,
                request.target_language,
                request.source_language,
                request.model.value,
                request.system_instruction,
                request.prompt_template
            )
    
    # Execute all batches
    tasks = [process_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results
    for result in results:
        if isinstance(result, Exception):
            all_errors.append(str(result))
        else:
            translations, errors = result
            all_translations.extend(translations)
            all_errors.extend(errors)
    
    # Sort translations by index
    all_translations.sort(key=lambda x: x.index)
    
    successful = len(all_translations)
    failed = len(request.lines) - successful
    
    return MultiTranslateResponse(
        translations=all_translations,
        total_lines=len(request.lines),
        successful_lines=successful,
        failed_lines=failed,
        errors=all_errors if all_errors else None
    )
