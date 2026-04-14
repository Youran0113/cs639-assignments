import pickle


VOCAB_PATH = "tokenizer_results/TinyStoriesV2-GPT4-train_vocab.pkl"
MERGES_PATH = "tokenizer_results/TinyStoriesV2-GPT4-train_merges.pkl"
INPUT_PATH = "tiny_test.txt"


def load_artifacts():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    with open(MERGES_PATH, "rb") as f:
        merges = pickle.load(f)

    return vocab, merges


def encode(text, merges):
    # Start with raw bytes
    tokens = [bytes([b]) for b in text.encode("utf-8")]

    merge_ranks = {pair: i for i, pair in enumerate(merges)}

    while True:
        pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

        candidates = [
            (merge_ranks[pair], i)
            for i, pair in enumerate(pairs)
            if pair in merge_ranks
        ]

        if not candidates:
            break

        _, idx = min(candidates)
        merged = tokens[idx] + tokens[idx + 1]

        tokens = tokens[:idx] + [merged] + tokens[idx + 2 :]

    return tokens


def decode(tokens):
    return b"".join(tokens).decode("utf-8", errors="ignore")


def main():
    vocab, merges = load_artifacts()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    print("=== ORIGINAL TEXT ===")
    print(text)

    tokens = encode(text, merges)

    print("\n=== ENCODED TOKENS ===")
    for t in tokens:
        try:
            print(t.decode("utf-8"))
        except:
            print(t)

    print("\nNumber of tokens:", len(tokens))

    decoded = decode(tokens)

    print("\n=== DECODED TEXT ===")
    print(decoded)

    print("\n=== VALIDATION ===")
    if decoded == text:
        print("✓ Round-trip is PERFECT (correct BPE behavior)")
    else:
        print("✗ Mismatch detected")


if __name__ == "__main__":
    main()