"""LLM client wrapper supporting Gemini and OpenAI."""

from typing import List, Dict, Optional
from nl2data.config.settings import get_settings
from nl2data.config.logging import get_logger

logger = get_logger(__name__)

# Global client instances
_gemini_client: Optional[object] = None
_openai_client: Optional[object] = None

# Global forced provider (None = use priority, "openai"/"local"/"gemini" = force that provider)
_forced_provider: Optional[str] = None


def _get_gemini_client():
    """Get or create the global Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        try:
            import google.generativeai as genai
            settings = get_settings()
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set in environment")
            genai.configure(api_key=settings.gemini_api_key)
            _gemini_client = genai
            logger.debug("Initialized Gemini client")
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
    return _gemini_client


def _get_openai_client():
    """Get or create the global OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            settings = get_settings()
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set in environment")
            _openai_client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.llm_timeout
            )
            logger.debug(f"Initialized OpenAI client with timeout={settings.llm_timeout}s")
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
    return _openai_client


def set_forced_provider(provider: Optional[str]) -> None:
    """
    Force a specific LLM provider to be used.
    
    Args:
        provider: "openai", "local", "gemini", or None to use priority order
    """
    global _forced_provider
    _forced_provider = provider
    logger.info(f"Forced LLM provider set to: {provider}")


def chat(messages: List[Dict[str, str]]) -> str:
    """
    Send messages to LLM API (Gemini, OpenAI, or local OpenAI-compatible).

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Content of the assistant's response

    Raises:
        Exception: If API call fails
    """
    global _forced_provider
    settings = get_settings()

    # If a provider is forced, use it
    if _forced_provider == "openai":
        return _chat_openai(messages)
    elif _forced_provider == "local":
        return _chat_local(messages)
    elif _forced_provider == "gemini":
        return _chat_gemini(messages)

    # Otherwise, use priority order (OpenAI > Gemini > Local)
    # Try each provider and fall back if it fails
    use_openai = settings.openai_api_key and settings.model_name
    use_gemini = settings.gemini_api_key and settings.gemini_model
    use_local = settings.llm_url and settings.model

    if not use_openai and not use_local and not use_gemini:
        raise ValueError(
            "No LLM API configured. Set either OPENAI_API_KEY/MODEL_NAME, "
            "LLM_URL/MODEL, or GEMINI_API_KEY/GEMINI_MODEL in .env"
        )

    # Try OpenAI first (if configured)
    if use_openai:
        try:
            return _chat_openai(messages)
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}. Falling back to next provider...")
            # Fall through to try next provider
    
    # Try Gemini second (if configured)
    if use_gemini:
        try:
            return _chat_gemini(messages)
        except Exception as e:
            logger.warning(f"Gemini call failed: {e}. Falling back to next provider...")
            # Fall through to try next provider
    
    # Try Local LLM last (if configured)
    if use_local:
        try:
            return _chat_local(messages)
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
            raise
    
    # If we get here, all configured providers failed
    raise RuntimeError(
        "All configured LLM providers failed. "
        "Please check your API keys and network connection."
    )


def _chat_gemini(messages: List[Dict[str, str]]) -> str:
    """Send messages to Gemini API."""
    import google.generativeai as genai
    import time
    settings = get_settings()
    genai_client = _get_gemini_client()

    logger.debug(
        f"Sending chat request to Gemini {settings.gemini_model} "
        f"(temperature={settings.temperature}, timeout={settings.llm_timeout}s)"
    )

    # Retry logic with exponential backoff for timeout errors
    max_retries = settings.llm_max_retries
    base_retry_delay = settings.llm_retry_delay
    
    for attempt in range(max_retries):
        try:
            # Convert messages format for Gemini
            # Gemini can handle chat history, but for simplicity we'll combine system + user
            system_content = ""
            user_parts = []
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    system_content = content
                elif role == "user":
                    user_parts.append(content)
                elif role == "assistant":
                    # For multi-turn conversations, we'd use chat history
                    # For now, we'll just use the last user message
                    pass

            # Combine system and user content
            if system_content:
                # Include system prompt in the user message for Gemini
                full_prompt = f"{system_content}\n\n{user_parts[-1] if user_parts else ''}"
            else:
                full_prompt = user_parts[-1] if user_parts else ""

            # Get the model with generation config
            model = genai.GenerativeModel(
                model_name=settings.gemini_model,
            )
            
            # Set generation config
            generation_config = {
                "temperature": settings.temperature,
            }

            response = model.generate_content(
                full_prompt,
                generation_config=generation_config,
                request_options={"timeout": settings.llm_timeout}
            )
            
            # Handle response - Gemini may return text or parts
            if hasattr(response, 'text') and response.text:
                content = response.text
            elif hasattr(response, 'parts') and response.parts:
                content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                content = str(response)
                
            logger.debug(f"Received response ({len(content)} chars)")
            return content
        except (TimeoutError, Exception) as e:
            error_str = str(e)
            # Check if it's a timeout or transient error
            is_timeout = (
                "timeout" in error_str.lower() or
                "timed out" in error_str.lower() or
                isinstance(e, TimeoutError)
            )
            is_transient = (
                is_timeout or
                "503" in error_str or
                "504" in error_str or
                "408" in error_str
            )
            
            if is_transient and attempt < max_retries - 1:
                retry_delay = base_retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"{'Timeout' if is_timeout else 'Transient'} error on attempt {attempt + 1}/{max_retries}: {error_str}. "
                    f"Retrying in {retry_delay:.1f} seconds... (timeout={settings.llm_timeout}s)"
                )
                time.sleep(retry_delay)
                continue
            else:
                logger.error(f"Gemini API call failed: {e}", exc_info=True)
                raise


def _chat_local(messages: List[Dict[str, str]]) -> str:
    """Send messages to local OpenAI-compatible API."""
    settings = get_settings()
    
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. "
            "Install with: pip install openai"
        )
    
    # Create client with custom base URL
    # Ensure base_url ends with /v1 if it doesn't already
    base_url = settings.llm_url.rstrip('/')
    if not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"
    
    client = OpenAI(
        base_url=base_url,
        api_key="not-needed",  # Local APIs often don't require a real key
        timeout=settings.llm_timeout
    )

    logger.debug(
        f"Sending chat request to local model {settings.model} at {settings.llm_url} "
        f"(temperature={settings.temperature}, timeout={settings.llm_timeout}s)"
    )

    from openai import APITimeoutError
    from nl2data.agents.tools.retry import retry_with_backoff
    
    def _make_request():
        response = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=settings.temperature,
        )
        
        # Handle response - check if choices exist and have content
        if not response.choices or len(response.choices) == 0:
            raise ValueError("No choices in response from local LLM")
        
        choice = response.choices[0]
        if not hasattr(choice, 'message') or choice.message is None:
            raise ValueError("No message in response choice from local LLM")
        
        content = choice.message.content
        if content is None:
            raise ValueError("No content in message from local LLM")
        
        logger.debug(f"Received response ({len(content)} chars)")
        return content
    
    return retry_with_backoff(
        func=_make_request,
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_retry_delay,
        timeout_errors=(APITimeoutError, TimeoutError),
        operation_name=f"Local LLM API call to {settings.model}",
    )


def _chat_openai(messages: List[Dict[str, str]]) -> str:
    """Send messages to OpenAI API."""
    settings = get_settings()
    client = _get_openai_client()

    logger.debug(
        f"Sending chat request to OpenAI {settings.model_name} "
        f"(temperature={settings.temperature}, timeout={settings.llm_timeout}s)"
    )

    from openai import APITimeoutError
    from nl2data.agents.tools.retry import retry_with_backoff
    
    def _make_request():
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            temperature=settings.temperature,
        )
        content = response.choices[0].message.content
        logger.debug(f"Received response ({len(content)} chars)")
        return content
    
    return retry_with_backoff(
        func=_make_request,
        max_retries=settings.llm_max_retries,
        base_delay=settings.llm_retry_delay,
        timeout_errors=(APITimeoutError, TimeoutError),
        operation_name=f"OpenAI API call to {settings.model_name}",
    )


def test_llm_availability() -> Optional[str]:
    """
    Test which LLM provider is available by trying each one with a small demo query.
    
    Priority order: OpenAI (GPT-4o) → Gemini → Local LLM
    
    Returns:
        Provider name ("openai", "gemini", "local") if available, None if none work
    """
    import time
    settings = get_settings()
    demo_messages = [
        {"role": "user", "content": "Say 'OK' if you can read this."}
    ]
    
    # Test OpenAI first (specifically GPT-4o or any GPT-4 model)
    if settings.openai_api_key:
        # Use gpt-4o as default if MODEL_NAME is not set, otherwise use configured model
        test_model = settings.model_name or "gpt-4o"
        model_lower = test_model.lower()
        
        # Only test if it's a GPT-4 model (gpt-4, 4o, o1, etc.)
        if "gpt-4" in model_lower or "4o" in model_lower or "o1" in model_lower or not settings.model_name:
            try:
                print(f"  Testing OpenAI with model: {test_model}...", end=" ", flush=True)
                # Temporarily set model_name for testing
                original_model_name = settings.model_name
                settings.model_name = test_model
                # Use a shorter timeout for the test (30 seconds)
                original_timeout = settings.llm_timeout
                settings.llm_timeout = 30.0
                try:
                    response = _chat_openai(demo_messages)
                    if response and len(response) > 0:
                        print("[OK]")
                        logger.info(f"[OK] OpenAI ({test_model}) is available")
                        # Keep model_name set if it wasn't set before
                        if not original_model_name:
                            # Don't restore, keep the test_model
                            pass
                        else:
                            settings.model_name = original_model_name
                        settings.llm_timeout = original_timeout
                        return "openai"
                finally:
                    # Only restore if test failed
                    if settings.model_name == test_model and original_model_name is not None:
                        settings.model_name = original_model_name
                    settings.llm_timeout = original_timeout
            except Exception as e:
                print(f"[FAILED]")
                logger.warning(f"OpenAI test failed: {e}")
    
    # Test Gemini
    if settings.gemini_api_key and settings.gemini_model:
        try:
            print(f"  Testing Gemini with model: {settings.gemini_model}...", end=" ", flush=True)
            # Use a shorter timeout for the test (30 seconds)
            original_timeout = settings.llm_timeout
            settings.llm_timeout = 30.0
            try:
                response = _chat_gemini(demo_messages)
                if response and len(response) > 0:
                    print("[OK]")
                    logger.info(f"[OK] Gemini ({settings.gemini_model}) is available")
                    return "gemini"
            finally:
                settings.llm_timeout = original_timeout
        except Exception as e:
            print(f"[FAILED]")
            logger.warning(f"Gemini test failed: {e}")
    
    # Test Local LLM
    if settings.llm_url and settings.model:
        try:
            print(f"  Testing Local LLM at {settings.llm_url} with model: {settings.model}...", end=" ", flush=True)
            # Use a shorter timeout for the test (30 seconds)
            original_timeout = settings.llm_timeout
            settings.llm_timeout = 30.0
            try:
                response = _chat_local(demo_messages)
                if response and len(response) > 0:
                    print("[OK]")
                    logger.info(f"[OK] Local LLM ({settings.model}) is available")
                    return "local"
            finally:
                settings.llm_timeout = original_timeout
        except Exception as e:
            print(f"[FAILED]")
            logger.warning(f"Local LLM test failed: {e}")
    
    print("\n[ERROR] No LLM providers are available")
    logger.error("No LLM providers are available")
    return None

