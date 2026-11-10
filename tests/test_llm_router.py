import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv

if importlib.util.find_spec("huggingface_hub") is None:
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.InferenceClient = type(
        "InferenceClient", (), {"__init__": lambda self, *args, **kwargs: None}
    )
    sys.modules["huggingface_hub"] = huggingface_hub

if importlib.util.find_spec("together") is None:
    together = types.ModuleType("together")
    together.Together = object
    together_error = types.ModuleType("together.error")
    together_error.RateLimitError = type("RateLimitError", (Exception,), {})
    together_error.ServiceUnavailableError = type(
        "ServiceUnavailableError", (Exception,), {}
    )
    sys.modules["together"] = together
    sys.modules["together.error"] = together_error

with patch.dict(
    os.environ,
    {"HF_TOKEN": "", "HF_ENDPOINT_URL": "", "TOGETHER_API_KEY": ""},
):
    UnifiedLLM = importlib.import_module("llm_router").UnifiedLLM


class _AlwaysFailingClient:
    def __init__(self):
        self.calls = 0
        self.errors = [RuntimeError(f"simulated provider failure {i}") for i in range(3)]

    def chat_completion(self, **_kwargs):
        error = self.errors[self.calls]
        self.calls += 1
        raise error


class UnifiedLLMTests(unittest.TestCase):
    def test_hf_chat_sleeps_only_between_attempts(self):
        provider = _AlwaysFailingClient()
        client = UnifiedLLM.__new__(UnifiedLLM)
        client.hf_client = provider

        with patch("llm_router.time.sleep") as sleep:
            with self.assertRaises(RuntimeError) as raised:
                client._hf_chat([{"role": "user", "content": "local fixture"}])

        self.assertEqual(provider.calls, 3)
        self.assertIs(raised.exception, provider.errors[-1])
        waits = [call.args[0] for call in sleep.call_args_list]
        self.assertEqual(len(waits), 2)
        self.assertAlmostEqual(waits[0], 2.2)
        self.assertAlmostEqual(waits[1], 3.96)


if __name__ == "__main__":
    unittest.main()
