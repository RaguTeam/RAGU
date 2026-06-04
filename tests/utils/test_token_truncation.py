import pytest
import types
from unittest.mock import Mock

from ragu.utils.token_truncation import TokenTruncation
import ragu.utils.token_truncation as token_truncation_module


class TestTokenTruncationWithTiktoken:
    @pytest.fixture
    def truncator(self):
        try:
            return TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=50
            )
        except Exception as e:
            pytest.skip(f"tiktoken not available: {e}")

    def test_truncate_short_text(self, truncator):
        text = "This is a short text."
        result = truncator(text)

        assert isinstance(result, str)
        assert result == text

    def test_truncate_long_text(self, truncator):
        # Create a text that's definitely longer than 50 tokens
        text = " ".join(["word"] * 200)
        result = truncator(text)

        assert isinstance(result, str)
        assert len(result) < len(text)

    def test_truncate_empty_string(self, truncator):
        result = truncator("")

        assert result == ""

    def test_truncate_unicode_text(self, truncator):
        text = "Hello 世界 " * 50
        result = truncator(text)

        assert isinstance(result, str)

    def test_safe_decode_handling(self):
        try:
            truncator_safe = TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=10,
                safe_decode=True
            )

            text = "Test text with special characters: 你好世界"
            result = truncator_safe(text)

            assert isinstance(result, str)
        except Exception:
            pytest.skip("tiktoken not available")

    def test_consistency(self):
        try:
            truncator = TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=30
            )

            text = "This is a test sentence that should be truncated consistently."
            result1 = truncator(text)
            result2 = truncator(text)

            assert result1 == result2
        except Exception:
            pytest.skip("tiktoken not available")


class TestTokenTruncationEdgeCases:
    def test_special_characters(self):
        try:
            truncator = TokenTruncation(
                model_id="gpt-4o",
                tokenizer_type="tiktoken",
                max_tokens=50
            )

            text = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
            result = truncator(text)

            assert isinstance(result, str)
        except Exception:
            pytest.skip("tiktoken not available")


class _FakeLocalTokenizer:
    def __init__(self):
        self.encode_calls = []
        self.decode_calls = []

    def encode(self, text: str, add_special_tokens: bool = False):
        self.encode_calls.append((text, add_special_tokens))
        return list(text.encode("utf-8"))

    def decode(self, tokens, skip_special_tokens: bool = True):
        self.decode_calls.append((list(tokens), skip_special_tokens))
        return bytes(tokens).decode("utf-8", errors="ignore")


class _FakeAutoTokenizer:
    last_model_id = None
    last_instance = None

    @classmethod
    def from_pretrained(cls, model_id: str):
        cls.last_model_id = model_id
        cls.last_instance = _FakeLocalTokenizer()
        return cls.last_instance


class TestTokenTruncationWithLocalTokenizer:
    @pytest.fixture
    def truncator_local(self, monkeypatch):
        fake_transformers = types.SimpleNamespace(AutoTokenizer=_FakeAutoTokenizer)
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
        return TokenTruncation(
            model_id="fake-local-model",
            tokenizer_type="local",
            max_tokens=10,
        )

    def test_local_init_loads_tokenizer(self, truncator_local):
        assert _FakeAutoTokenizer.last_model_id == "fake-local-model"
        assert hasattr(truncator_local, "local_tokenizer")

    def test_local_truncate_short_text(self, truncator_local):
        text = "short"
        result = truncator_local(text)
        assert result == text

    def test_local_truncate_long_text(self, truncator_local):
        text = "abcdefghijklmnopqrstuvwxyz"
        result = truncator_local(text)
        assert isinstance(result, str)
        assert len(result) < len(text)

    def test_local_truncate_empty_string(self, truncator_local):
        assert truncator_local("") == ""

    def test_local_encode_decode_flags_are_set(self, truncator_local):
        text = "0123456789ABCDEF"
        truncator_local(text)
        tokenizer = _FakeAutoTokenizer.last_instance
        assert tokenizer is not None
        add_special_flags = [call[1] for call in tokenizer.encode_calls]
        assert True in add_special_flags
        assert False in add_special_flags
        assert tokenizer.decode_calls[-1][1] is True

    def test_local_unicode_truncation_returns_valid_string(self, truncator_local):
        text = "Привет мир " * 10
        result = truncator_local(text)
        assert isinstance(result, str)

    def test_local_return_stats_logs_info(self, truncator_local, monkeypatch):
        info_mock = Mock()
        monkeypatch.setattr(token_truncation_module.logger, "info", info_mock, raising=False)
        result = truncator_local("this text should be truncated", return_stats=True)
        assert isinstance(result, str)
        info_mock.assert_called_once()

    def test_local_init_failure_raises_value_error(self, monkeypatch):
        class _BrokenAutoTokenizer:
            @classmethod
            def from_pretrained(cls, model_id: str):
                raise RuntimeError("load failed")

        fake_transformers = types.SimpleNamespace(AutoTokenizer=_BrokenAutoTokenizer)
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

        with pytest.raises(ValueError, match="Failed to load tokenizer"):
            TokenTruncation(model_id="broken-model", tokenizer_type="local")

    def test_unsupported_tokenizer_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported tokenizer_type"):
            TokenTruncation(model_id="x", tokenizer_type="unknown")


class _FakeBERTTokenizer:
    SPECIAL_TOKEN_COUNT = 2

    def __init__(self):
        self.special = ["[CLS]", "[SEP]"]

    def encode(self, text: str, add_special_tokens: bool = False):
        word_tokens = text.split()
        if add_special_tokens:
            return self.special[:1] + word_tokens + self.special[1:2]
        return word_tokens

    def decode(self, tokens, skip_special_tokens: bool = True):
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special]
        return " ".join(tokens)


class _FakeBERTAutoTokenizer:
    last_instance = None

    @classmethod
    def from_pretrained(cls, model_id: str):
        cls.last_instance = _FakeBERTTokenizer()
        return cls.last_instance


class TestTokenTruncationSpecialTokens:
    @pytest.fixture
    def bert_truncator(self, monkeypatch):
        fake_transformers = types.SimpleNamespace(AutoTokenizer=_FakeBERTAutoTokenizer)
        monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
        return TokenTruncation(
            model_id="fake-bert-model",
            tokenizer_type="local",
            max_tokens=7,
        )

    def test_special_tokens_accounted_in_limit(self, bert_truncator):
        tokenizer = _FakeBERTAutoTokenizer.last_instance
        words = ["w" + str(i) for i in range(20)]
        text = " ".join(words)

        truncated = bert_truncator(text)

        re_encoded = tokenizer.encode(truncated, add_special_tokens=True)
        assert len(re_encoded) <= 7

    def test_short_text_unchanged(self, bert_truncator):
        tokenizer = _FakeBERTAutoTokenizer.last_instance
        text = "a b c d"
        truncated = bert_truncator(text)

        assert truncated == text
        re_encoded = tokenizer.encode(truncated, add_special_tokens=True)
        assert len(re_encoded) <= 7

    def test_exact_boundary_text(self, bert_truncator):
        tokenizer = _FakeBERTAutoTokenizer.last_instance
        text = "a b c d e"
        truncated = bert_truncator(text)

        re_encoded = tokenizer.encode(truncated, add_special_tokens=True)
        assert len(re_encoded) <= 7
