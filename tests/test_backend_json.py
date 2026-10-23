"""Controller stopping behavior without model weights, CUDA or network."""

import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from clipscout.backends import QwenBackend, _json_stopping_criteria, json_object_complete


class JsonStoppingTests(unittest.TestCase):
    def test_stops_at_first_complete_object_during_incremental_generation(self):
        first = '{"tool":"search","args":{"query":"entropy","top_k":5}}'
        continuation = first + '\n{"tool":"inspect_clip","args":{"start":120,"end":150}}'
        emitted = ""
        for character in continuation:
            emitted += character
            if json_object_complete(emitted):
                break
        self.assertEqual(emitted, first)
        self.assertEqual(json.loads(emitted)["tool"], "search")

    def test_quotes_escapes_nested_arrays_and_unicode_do_not_stop_early(self):
        objects = [
            {"answer": 'A literal } and { with a quote " and backslash \\ inside text.'},
            {"answer": "Закрывающая скобка } внутри строки", "citations": [{"evidence_id": "E001", "start": 1, "end": 2}]},
            {"args": {"values": [[{"text": "\\\"}"}], [], {}]}},
        ]
        for value in objects:
            text = json.dumps(value, ensure_ascii=False)
            with self.subTest(text=text):
                for end in range(len(text)):
                    self.assertFalse(json_object_complete(text[:end]), text[:end])
                self.assertTrue(json_object_complete(text))
                self.assertTrue(json_object_complete(" \n\t" + text))

    def test_incomplete_nonobject_and_mismatched_containers_do_not_trigger(self):
        for text in ("", "  ", "[]", "null", '{"a":', '{"a":"unterminated }',
                     '{"a":[}', 'preface {"a":1}', '```json\n{"a":1}'):
            with self.subTest(text=text):
                self.assertFalse(json_object_complete(text))

    def test_generation_criterion_ignores_prompt_json(self):
        class TokenIds:
            def __init__(self, characters):
                self.characters = characters

            def __getitem__(self, key):
                row, span = key
                self.last_row = row
                return self.characters[span]

        class Tokenizer:
            def decode(self, characters, **kwargs):
                return "".join(characters)

        fake_transformers = SimpleNamespace(StoppingCriteria=object, StoppingCriteriaList=list)
        prompt = '{"example":"this prompt object must not end generation"}'
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            criterion = _json_stopping_criteria(Tokenizer(), len(prompt))[0]
        self.assertFalse(criterion(TokenIds(prompt + '{"tool":'), None))
        self.assertTrue(criterion(TokenIds(prompt + '{"tool":"search"}'), None))

    def test_action_stop_is_opt_in_and_observation_generation_is_unchanged(self):
        backend = QwenBackend(device="cpu", local_files_only=True)
        with patch.object(backend, "_generate", return_value="returned unchanged") as generate:
            self.assertEqual(backend.generate_action("system", "user"), "returned unchanged")
            self.assertTrue(generate.call_args.kwargs["stop_json"])
            self.assertIn("one JSON object", generate.call_args.args[0])
            generate.reset_mock()
            images = [(1.0, object())]
            backend.generate("system", "user", images)
            self.assertEqual(generate.call_args.args, ("system", "user", images))
            self.assertNotIn("stop_json", generate.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
