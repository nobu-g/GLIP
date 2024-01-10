def align_chunks(chunks_a: list[int], chunks_b: list[int]) -> list[list[int]]:
    assert sum(chunks_a) == sum(chunks_b)
    # assert all(chunk > 0 for chunk in chunks_a), "The length of each chunk must be positive."
    assert all(chunk > 0 for chunk in chunks_b), "The length of each chunk must be positive."
    alignments: list[list[int]] = []
    item_to_chunk_b: list[int] = []
    for i, chunk_b in enumerate(chunks_b):
        item_to_chunk_b += [i for _ in range(chunk_b)]
    cursor = 0
    for chunk_a in chunks_a:
        if chunk_a == 0:
            alignments.append([])
            continue
        aligned_chunks = sum(alignments, [])
        related_chunks = set()
        for i in range(cursor, cursor + chunk_a):
            related_chunks.add(item_to_chunk_b[i])
        if len(related_chunks) > 1:
            for item in aligned_chunks:
                if item in related_chunks:
                    related_chunks.remove(item)
        assert len(related_chunks) >= 1
        alignments.append(sorted(related_chunks))
        cursor += chunk_a
    assert len(alignments) == len(chunks_a), f"{len(alignments)} != {len(chunks_a)}"
    return alignments


def test_chunk_aligner():
    cases = [
        {
            "chunk_a": [2, 1, 2],
            "chunk_b": [2, 2, 1],
            "alignments": [[0], [1], [2]],
        },
        {
            "chunk_a": [0, 0, 2, 1, 2],
            "chunk_b": [2, 2, 1],
            "alignments": [[], [], [0], [1], [2]],
        },
        {
            "chunk_a": [4, 1],
            "chunk_b": [2, 2, 1],
            "alignments": [[0, 1], [2]],
        },
        {
            "chunk_a": [1, 1, 1, 1, 1],
            "chunk_b": [2, 2, 1],
            "alignments": [[0], [0], [1], [1], [2]],
        },
        {
            "chunk_a": [3, 2],
            "chunk_b": [2, 2, 1],
            "alignments": [[0, 1], [2]],
        },
        {
            "chunk_a": [3, 1, 1],
            "chunk_b": [2, 2, 1],
            "alignments": [[0, 1], [1], [2]],  # TODO: [0], [1], [2]
        },
    ]
    for case in cases:
        chunks_a = case["chunk_a"]
        chunks_b = case["chunk_b"]
        expected = case["alignments"]
        actual = align_chunks(chunks_a, chunks_b)
        assert actual == expected, f"{actual} != {expected}"


if __name__ == "__main__":
    test_chunk_aligner()
