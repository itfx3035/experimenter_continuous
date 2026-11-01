import unittest

from experimenter.parsing import extract_code, extract_json, parse_json_output


class ParseJsonTests(unittest.TestCase):
    def test_first_object_with_prose_and_fences(self):
        text = 'Sure!\n```json\n{"score": 0.5, "details": "ok"}\n```\nHope that helps.'
        self.assertEqual(extract_json(text), {"score": 0.5, "details": "ok"})

    def test_nested_braces_inside_strings_are_ignored(self):
        data = extract_json('{"details": "a } brace in a string {", "score": 0.25}')
        self.assertEqual(data["score"], 0.25)

    def test_escaped_quotes(self):
        data = extract_json('{"details": "quote \\" inside", "score": 0.1}')
        self.assertEqual(data["details"], 'quote " inside')

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json("no json here")

    def test_unclosed_json_raises(self):
        with self.assertRaises(ValueError):
            extract_json('{"score": 0.5')

    def test_last_json_line_is_used_for_output(self):
        out = '{"score": 0.1}\nnoise\n{"score": 0.9, "passed": true}\n'
        self.assertEqual(parse_json_output(out)["score"], 0.9)

    def test_pretty_printed_multi_line_json(self):
        out = "timing done\n{\n  \"score\": 0.7,\n  \"passed\": true\n}\n"
        self.assertEqual(parse_json_output(out), {"score": 0.7, "passed": True})

    def test_non_object_json_returns_none(self):
        # Callers do parsed.get("score") — a list would crash them.
        self.assertIsNone(parse_json_output('["not", "an", "object"]'))
        self.assertIsNone(parse_json_output(""))


class ParseCodeTests(unittest.TestCase):
    def test_code_fence(self):
        self.assertEqual(extract_code("here:\n```python\ndef f():\n    return 1\n```"),
                         "def f():\n    return 1")

    def test_plain_text_without_fence(self):
        self.assertEqual(extract_code("def f():\n    return 1"), "def f():\n    return 1")


if __name__ == "__main__":
    unittest.main()
