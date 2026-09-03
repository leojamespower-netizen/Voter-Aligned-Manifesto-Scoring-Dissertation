"""Single point of contact for every API call. Nothing else calls a model."""

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve paths relative to the repo root (parent of this package), so the code works regardless of the notebook's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "cache"
CALL_LOG = CACHE_DIR / "call_log.csv"

# Model identifiers used throughout the pipeline.
GPT5 = "gpt-5" # closed-weight, fixed temperature via OPENAI API
CLAUDE = "claude" # closed-weight, fixed temperature via the Anthropic API


# Registered decoding parameters, per stage.
# Temperature doesn't apply to GPT-5 due to fixed parameterisation
# It does however apply to Claude Opus 4.5, and as such is included for flexibility, even though it will be overriden on the OPENAI side.
DECODING = {
    "summaries":   {"temperature": 1.0,}, 
    "extractions": {"temperature": 1.0,},
    "profiles":    {"temperature": 1.0,},
    "comparisons": {"temperature": 1.0,},
    "probes":      {"temperature": 1.0}
}

DEFAULT_TEMPERATURE = 1.0

TEMPERATURE_OVERRIDE = None

def call_llm(prompt, model, cache_key, subdir='comparisons', system=None,
             temperature=None, max_retries=6, force_refresh=False):  
    """Cached API call. Returns the cached response if one exists.

    The cache key holds the identifiers of the inputs, not their content:
    edit an input and the stale response is still served. force_refresh is
    manual; deleting the subdirectory is safer.

    Decoding parameters come from DECODING, so the values sent to the API
    and the values written into the cache file are the same object. An
    earlier version took them from two places, and every file recorded a
    temperature the request had not used.
    """
    settings = DECODING.get(subdir, {})
    if temperature is None:
        temperature = (TEMPERATURE_OVERRIDE if TEMPERATURE_OVERRIDE is not None
                       else settings.get("temperature", DEFAULT_TEMPERATURE))

    decoding = {}  # exactly what is sent to the API; also written to the cache file
    if temperature != DEFAULT_TEMPERATURE:
        decoding["temperature"] = temperature

    if "max_completion_tokens" in settings:
        decoding["max_completion_tokens"] = settings["max_completion_tokens"]

    if PROVIDERS[model]["sdk"] == "anthropic":
        decoding["max_tokens"] = ANTHROPIC_MAX_TOKENS  # required by the Messages API

    # the key carries the temperature so that runs at different values
    # write to different files rather than one serving the other's responses
    if temperature != DEFAULT_TEMPERATURE:
        cache_key = f"{cache_key}_t{temperature}"
    
    cache_file = CACHE_DIR / subdir / f"{cache_key}.json"

    if cache_file.exists() and not force_refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            text = _api_request(prompt, model, system, decoding)
            break
        except Exception as exc:  # noqa: BLE001 - want to retry on any API error
            last_error = exc
            wait = min(2 ** attempt, 60)  # backs off up to a minute; an overnight run dropped connection
            print(f"[llm_client] attempt {attempt} failed for {cache_key}: {exc}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"All {max_retries} attempts failed for {cache_key}"
        ) from last_error

    response = {
        "text": text,
        "model": model,
        "cache_key": cache_key,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "system": system,
        # Ensures a divergence found at write-up can attributed to inputs, prompt, or model version rather than estimated.
        "api_model": PROVIDERS[model]["api_model"],
        "decoding": decoding,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
    }

    # This is the money-saving line.
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(response, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    _log_call(cache_key, model, subdir)
    return response


# providers, keyed by model id
# Opus 4.1 was retired on 5 August 2026; Opus 4.5 is the nearest surviving model in tier and date
PROVIDERS = {
    GPT5: {"sdk": "openai", "key_var": "OPENAI_API_KEY", "api_model": "gpt-5"},
    CLAUDE: {"sdk": "anthropic", "key_var": "ANTHROPIC_API_KEY", "api_model": "claude-opus-4-5-20251101"},
}

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 16000  # mandatory field, set far above the longest response seen (~2,500 tokens)
_CLIENTS = {}


def _api_key(model):
    """Returns the provider's API key from the environment."""
    spec = PROVIDERS[model]
    key = os.environ.get(spec["key_var"])
    if not key:
        raise RuntimeError(
            f"{spec['key_var']} is not set. Put it in .env (gitignored) or "
            f"export it before running.")
    return key


def _client(model):
    """Build and cache one OpenAI client per model."""
    if model in _CLIENTS:
        return _CLIENTS[model]
    key = _api_key(model)  # checked before the import so a missing key is reported as such
    from openai import OpenAI
    _CLIENTS[model] = OpenAI(api_key=key)
    return _CLIENTS[model]

def _anthropic_post(body, key):
    """Posts one request to the Messages API and returns the parsed response.
    Called over HTTP because the anthropic SDK (v1.0+) refuses the temperature parameter."""
    import httpx
    response = httpx.post(
        ANTHROPIC_URL,
        headers={"x-api-key": key,
                 "anthropic-version": ANTHROPIC_VERSION,
                 "content-type": "application/json"},
        json=body,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _api_request(prompt, model, system, decoding):
    """Send one prompt and return the response text.

    Args:
        prompt (str): the user message.
        model (str): a key of PROVIDERS.
        system (str): optional system message.
        decoding (dict): decoding parameters, in the API's own names. Empty means defaults.

    Returns:
        str: the model's reply.
    """
    if model not in PROVIDERS:
        raise ValueError(f"Unknown model: {model}")
    spec = PROVIDERS[model]

    if spec["sdk"] == "openai":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = _client(model).chat.completions.create(
            model=spec["api_model"],
            messages=messages,
            **decoding,
        )
        return response.choices[0].message.content or ""

    if spec["sdk"] == "anthropic":
        body = {"model": spec["api_model"],
                "messages": [{"role": "user", "content": [{"type: "text", "text":prompt,
                "cache_control: {"type": "ephemeral"}}]}],
                **decoding}
        if system:
            body["system"] = system  # top-level on this API, not a message
        data = _anthropic_post(body, _api_key(model))
        return "".join(block.get("text", "") for block in data.get("content", [])
                       if block.get("type") == "text")

    raise ValueError(f"Unknown sdk for {model}: {spec['sdk']}")

# append one row per fresh API call to cache/call_log.csv
def _log_call(cache_key, model, subdir):
    CALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CALL_LOG.exists()
    with CALL_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp_utc", "cache_key", "model", "subdir"])
        writer.writerow([datetime.now(timezone.utc).isoformat(),
                         cache_key, model, subdir])


# load all cached responses in cache/{subdir} matching `pattern`
def load_cached(subdir, pattern='*.json'):
    folder = CACHE_DIR / subdir
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No cached responses found in {folder} matching '{pattern}'. "
            f"Has the cache/ folder been committed and cloned?"
        )
    return [json.loads(p.read_text(encoding="utf-8")) for p in files]