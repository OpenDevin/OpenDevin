import pytest

from openhands.utils.chunk_localizer import (
    Chunk,
    create_chunks,
    get_top_k_chunk_matches,
    normalized_lcs,
)


def test_chunk_creation():
    chunk = Chunk(text='test chunk', line_range=(1, 1))
    assert chunk.text == 'test chunk'
    assert chunk.line_range == (1, 1)
    assert chunk.normalized_lcs is None


def test_chunk_visualization(capsys):
    chunk = Chunk(text='line1\nline2', line_range=(1, 2))
    assert chunk.visualize() == '1|line1\n2|line2\n'


def test_create_chunks_raw_string():
    text = 'line1\nline2\nline3\nline4\nline5'
    chunks = create_chunks(text, size=2)
    assert len(chunks) == 3
    assert chunks[0].text == 'line1\nline2'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3\nline4'
    assert chunks[1].line_range == (3, 4)
    assert chunks[2].text == 'line5'
    assert chunks[2].line_range == (5, 6)


def test_normalized_lcs():
    chunk = 'abcdef'
    edit_draft = 'abcxyz'
    assert normalized_lcs(chunk, edit_draft) == 0.5


def test_get_top_k_chunk_matches():
    text = 'chunk1\nchunk2\nchunk3\nchunk4'
    query = 'chunk2'
    matches = get_top_k_chunk_matches(text, query, k=2, max_chunk_size=1)
    assert len(matches) == 2
    assert matches[0].text == 'chunk2'
    assert matches[0].line_range == (2, 2)
    assert matches[0].normalized_lcs == 1.0
    assert matches[1].text == 'chunk1'
    assert matches[1].line_range == (1, 1)
    assert matches[1].normalized_lcs == 5 / 6
    assert matches[0].normalized_lcs > matches[1].normalized_lcs


def test_create_chunks_with_empty_lines():
    text = 'line1\n\nline3\n\n\nline6'
    chunks = create_chunks(text, size=2)
    assert len(chunks) == 3
    assert chunks[0].text == 'line1\n'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3\n'
    assert chunks[1].line_range == (3, 4)
    assert chunks[2].text == '\nline6'
    assert chunks[2].line_range == (5, 6)


def test_create_chunks_with_large_size():
    text = 'line1\nline2\nline3'
    chunks = create_chunks(text, size=10)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].line_range == (1, 3)


def test_create_chunks_with_last_chunk_smaller():
    text = 'line1\nline2\nline3'
    chunks = create_chunks(text, size=2)
    assert len(chunks) == 2
    assert chunks[0].text == 'line1\nline2'
    assert chunks[0].line_range == (1, 2)
    assert chunks[1].text == 'line3'
    assert chunks[1].line_range == (3, 3)


def test_normalized_lcs_edge_cases():
    assert normalized_lcs('', '') == 0.0
    assert normalized_lcs('a', '') == 0.0
    assert normalized_lcs('', 'a') == 0.0
    assert normalized_lcs('abcde', 'ace') == 0.6


def test_get_top_k_chunk_matches_with_ties():
    text = 'chunk1\nchunk2\nchunk3\nchunk1'
    query = 'chunk'
    matches = get_top_k_chunk_matches(text, query, k=3, max_chunk_size=1)
    assert len(matches) == 3
    assert all(match.normalized_lcs == 5 / 6 for match in matches)
    assert {match.text for match in matches} == {'chunk1', 'chunk2', 'chunk3'}


def test_get_top_k_chunk_matches_with_large_k():
    text = 'chunk1\nchunk2\nchunk3'
    query = 'chunk'
    matches = get_top_k_chunk_matches(text, query, k=10, max_chunk_size=1)
    assert len(matches) == 3  # Should return all chunks even if k is larger


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 4])
def test_create_chunks_different_sizes(chunk_size):
    text = 'line1\nline2\nline3\nline4'
    chunks = create_chunks(text, size=chunk_size)
    assert len(chunks) == (4 + chunk_size - 1) // chunk_size
    assert sum(len(chunk.text.split('\n')) for chunk in chunks) == 4


def test_chunk_visualization_with_special_characters():
    chunk = Chunk(text='line1\nline2\t\nline3\r', line_range=(1, 3))
    assert chunk.visualize() == '1|line1\n2|line2\t\n3|line3\r\n'


def test_normalized_lcs_with_unicode():
    chunk = 'Hello, 世界!'
    edit_draft = 'Hello, world!'
    assert 0 < normalized_lcs(chunk, edit_draft) < 1


def test_get_top_k_chunk_matches_with_overlapping_chunks():
    text = 'chunk1\nchunk2\nchunk3\nchunk4'
    query = 'chunk2\nchunk3'
    matches = get_top_k_chunk_matches(text, query, k=2, max_chunk_size=2)
    assert len(matches) == 2
    assert matches[0].text == 'chunk1\nchunk2'
    assert matches[0].line_range == (1, 2)
    assert matches[1].text == 'chunk3\nchunk4'
    assert matches[1].line_range == (3, 4)
    assert matches[0].normalized_lcs == matches[1].normalized_lcs
