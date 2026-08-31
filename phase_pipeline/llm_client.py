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
GPT5 = "gpt-5" # closed-weight, via OpenAI API

# Registered decoding parameters, per stage.
DECODING = {
    "summaries":   {"temperature": 0.0, "max_tokens": 1200},
    "profiles":    {"temperature": 0.0, "max_tokens": 1500},
    "comparisons": {"temperature": 0.0, "max_tokens": 800},
    "probes":      {"temperature": 0.0, "max_tokens": 400},
}


def call_llm(prompt, model, cache_key, subdir='comparisons', system=None,
             temperature=1.0, max_retries=3, force_refresh=False):
    """Cached API call. Returns the cached response if one exists.

    The cache key holds the identifiers of the inputs, not their content:
    edit an input and the stale response is still served. force_refresh is
    manual; deleting the subdirectory is safer."""
    cache_file = CACHE_DIR / subdir / f"{cache_key}.json"

    if cache_file.exists() and not force_refresh:
        return json.loads(cache_file.read_text(encoding="utf-8"))

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            text = _api_request(prompt, model, system, temperature)
            break
        except Exception as exc:  # noqa: BLE001 - want to retry on any API error
            last_error = exc
            wait = 2 ** attempt 
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
        "decoding": DECODING.get(subdir, {"temperature": temperature}),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
    }

    # This is the money-saving line.
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(response, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    _log_call(cache_key, model, subdir)
    return response


# providers, keyed by model id
PROVIDERS = {
    GPT5: {"key_var": "OPENAI_API_KEY", "api_model": "gpt-5"},
}

_CLIENTS = {}


def _client(model):
    """Build and cache one API client per model.

    Args:
        model (str): a key of PROVIDERS.

    Returns:
        OpenAI: a configured client.

    Raises:
        RuntimeError: if the provider's key is not in the environment.
    """
    if model in _CLIENTS:
        return _CLIENTS[model]

    from openai import OpenAI

    spec = PROVIDERS[model]
    key = os.environ.get(spec["key_var"])
    if not key:
        raise RuntimeError(
            f"{spec['key_var']} is not set. Put it in .env (gitignored) or "
            f"export it before running.")

    _CLIENTS[model] = OpenAI(api_key=key)
    return _CLIENTS[model]


def _api_request(prompt, model, system, temperature):
    """Send one prompt and return the response text.

    Args:
        prompt (str): the user message.
        model (str): a key of PROVIDERS.
        system (str): optional system message.
        temperature (float): decoding temperature.

    Returns:
        str: the model's reply.
    """
    if model not in PROVIDERS:
        raise ValueError(f"Unknown model: {model}")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client(model).chat.completions.create(
        model=PROVIDERS[model]["api_model"],
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


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