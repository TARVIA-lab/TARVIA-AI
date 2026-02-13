from tarvia_ai.chunking import chunk_text


def test_chunk_text_overlap_and_size() -> None:
    text = "A" * 2500
    chunks = chunk_text(text=text, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 3
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 1000
    assert len(chunks[2]) == 900
