import tempfile
from pathlib import Path

import numpy as np

from ada.tokenizer import ADABPETokenizer, SPECIAL_TOKENS


def test_special_tokens_exist():
    tok = ADABPETokenizer()
    assert tok.pad_token_id is not None
    assert tok.bos_token_id is not None
    assert tok.eos_token_id is not None
    assert tok.unk_token_id is not None


def test_train_and_lossless_roundtrip():
    texts = [
        "The Adaptive Depth Architecture routes simple tokens through fewer layers.",
        "Mamba-2 SSD provides linear time state space recurrence.",
        "Mixture-of-Experts routes tokens to sparse expert feedforward networks.",
        "Special mathematical characters and symbols: \u2211 \u222b \u03c0 = 3.14159.",
    ]
    tok = ADABPETokenizer()
    tok.train(texts, vocab_size=320, min_frequency=1, show_progress=False)

    for text in texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        assert decoded == text, f"Decoded mismatch: '{decoded}' != '{text}'"


def test_encode_batch_and_numpy():
    texts = [
        "First document for testing.",
        "Second document with different words.",
    ]
    tok = ADABPETokenizer()
    tok.train(texts, vocab_size=300, min_frequency=1, show_progress=False)

    batch_ids = tok.encode_batch(texts, add_special_tokens=True)
    assert len(batch_ids) == 2
    assert batch_ids[0][0] == tok.bos_token_id
    assert batch_ids[0][-1] == tok.eos_token_id

    arr = tok.encode_to_numpy(texts, add_eos=True)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint32
    assert len(arr) > 0


def test_save_and_load():
    texts = ["Quick brown fox jumps over lazy dog."]
    tok = ADABPETokenizer()
    tok.train(texts, vocab_size=280, min_frequency=1, show_progress=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tokenizer.json"
        tok.save(save_path)
        assert save_path.exists()

        loaded = ADABPETokenizer.from_file(save_path)
        assert loaded.vocab_size == tok.vocab_size
        encoded_orig = tok.encode(texts[0])
        encoded_loaded = loaded.encode(texts[0])
        assert encoded_orig == encoded_loaded
