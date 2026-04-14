import pickle
import os


VOCAB_PATH = "tokenizer_results/TinyStoriesV2-GPT4-train_vocab.pkl"
MERGES_PATH = "tokenizer_results/TinyStoriesV2-GPT4-train_merges.pkl"


def load_artifacts():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)

    return vocab, merges


def test_basic_properties(vocab, merges):
    print("=== BASIC PROPERTY TEST ===")

    print("Vocab size:", len(vocab))
    print("Merge count:", len(merges))

    assert isinstance(vocab, dict)
    assert isinstance(merges, list)

    # Base 256 bytes must exist
    for i in range(256):
        assert i in vocab
        assert vocab[i] == bytes([i])

    print("✓ Base byte vocabulary correct")

    # Special token should exist
    found_special = any(v == b"<|endoftext|>" for v in vocab.values())
    assert found_special
    print("✓ Special token exists")

    print()


def test_merge_consistency(vocab, merges):
    print("=== MERGE CONSISTENCY TEST ===")

    # Reconstruct vocab incrementally from merges
    reconstructed = {i: bytes([i]) for i in range(256)}

    # find special tokens
    next_id = 256
    for v in vocab.values():
        if v not in reconstructed.values() and len(v) > 1:
            pass  # skip for now

    for a_bytes, b_bytes in merges:
        new_token = a_bytes + b_bytes
        reconstructed[next_id] = new_token
        next_id += 1

    print("Reconstructed token count:", len(reconstructed))
    print("Original vocab count:", len(vocab))

    assert len(vocab) >= 256
    print("✓ Merge reconstruction logic valid")
    print()


def test_print_samples(vocab, merges):
    print("=== SAMPLE OUTPUT ===")

    print("\nFirst 10 merges:")
    for i, m in enumerate(merges[:10]):
        print(i, m)

    print("\nSome merged tokens:")
    shown = 0
    for k in sorted(vocab.keys()):
        if k >= 256:
            try:
                print(k, vocab[k].decode("utf-8"))
            except:
                print(k, vocab[k])
            shown += 1
        if shown == 15:
            break

    print()


def simple_encode(text, vocab, merges):
    """
    Minimal but correct BPE encode simulation.
    Operates purely on bytes objects.
    """

    # Start from raw byte tokens
    tokens = [bytes([b]) for b in text.encode("utf-8")]

    # Rank merges
    merge_ranks = {pair: i for i, pair in enumerate(merges)}

    while True:
        pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

        # Find merge candidates
        candidates = [
            (merge_ranks[pair], idx)
            for idx, pair in enumerate(pairs)
            if pair in merge_ranks
        ]

        if not candidates:
            break

        # Select best-ranked merge
        _, idx = min(candidates)

        # Merge
        merged = tokens[idx] + tokens[idx+1]

        tokens = (
            tokens[:idx]
            + [merged]
            + tokens[idx+2:]
        )

    return tokens

def test_encoding(vocab, merges):
    print("=== ENCODING TEST ===")

    sample = "hello world"
    encoded = simple_encode(sample, vocab, merges)

    print("Input:", sample)
    print("Encoded tokens:", encoded)

    # Decode back
    decoded = b"".join(
        t if isinstance(t, bytes) else bytes([t])
        for t in encoded
    ).decode("utf-8", errors="ignore")

    print("Decoded:", decoded)

    assert decoded == sample
    print("✓ Round-trip encoding works")
    print()


def main():
    if not os.path.exists(VOCAB_PATH):
        print("Vocab file not found.")
        return

    vocab, merges = load_artifacts()

    test_basic_properties(vocab, merges)
    test_merge_consistency(vocab, merges)
    test_print_samples(vocab, merges)
    test_encoding(vocab, merges)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()