# src/app/endpoints/chat.py
import time
import asyncio
import re
from typing import List, Tuple
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
    source_language: str | None,
    system_instruction: str | None,
    prompt_template: str | None
) -> str:
    """Build the translation prompt for a batch of subtitle lines."""
    
    # Format lines as "index: content"
    lines_text = "\n".join([f"{line.index}: {line.content}" for line in lines])
    
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
5. Do not add any explanations or extra text
6. Return ONLY the translated lines in the same format

Input subtitles:
{lines_text}

Output (translated to {target_language}):"""
    
    return prompt


def _parse_translation_response(
    response_text: str,
    original_lines: List[SubtitleLine]
) -> List[TranslatedLine]:
    """Parse the translation response and extract translated lines."""
    
    translated = []
    original_map = {line.index: line.content for line in original_lines}
    
    # Try to parse lines in format "index: content"
    # Handle various formats the model might return
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to match "index: content" or "index. content" or just numbered lines
        match = re.match(r'^(\d+)[:\.\)]\s*(.+)$', line)
        if match:
            try:
                idx = int(match.group(1))
                content = match.group(2).strip()
                if idx in original_map:
                    translated.append(TranslatedLine(
                        index=idx,
                        original=original_map[idx],
                        translated=content
                    ))
            except ValueError:
                continue
    
    return translated


async def _translate_batch(
    gemini_client,
    lines: List[SubtitleLine],
    target_language: str,
    source_language: str | None,
    model_value: str,
    system_instruction: str | None,
    prompt_template: str | None
) -> Tuple[List[TranslatedLine], List[str]]:
    """Translate a batch of subtitle lines."""
    
    errors = []
    translated = []
    
    try:
        prompt = _build_translation_prompt(
            lines, target_language, source_language,
            system_instruction, prompt_template
        )
        
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
                
    except Exception as e:
        error_msg = f"Translation batch failed: {str(e)}"
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
