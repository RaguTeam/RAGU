import base64
from unittest.mock import AsyncMock

import pytest

from ragu.chunker.markdown_chunkers import RecursiveMarkdownChunker
from ragu.chunker.types import CodeChunk, Document, FormulaChunk, ImageChunk, TableChunk, TextChunk
from ragu.models.llm import LLM


@pytest.mark.parametrize(('object_text', 'object_type'), [
    ('$$E=mc^2$$', FormulaChunk),
    ('```python\nprint(1)\n```', CodeChunk),
    ('<table><tr><td>1</td></tr></table>', TableChunk),
    ('![plot](images/chart.png)', ImageChunk),
])
def test_text_stays_on_each_side_of_objects(object_text, object_type):
    chunks = RecursiveMarkdownChunker().split(f'Before.\n\n{object_text}\n\nAfter.')
    assert [type(chunk) for chunk in chunks] == [TextChunk, object_type, TextChunk]
    assert [chunks[0].content, chunks[2].content] == ['Before.', 'After.']
    assert [chunk.chunk_order_idx for chunk in chunks] == [0, 1, 2]


def test_adjacent_objects_and_sections_preserve_order():
    chunks = RecursiveMarkdownChunker().split('# First\n\n$$x$$\n\n$$y$$\n\n## Second\n\nEnd.')
    assert [type(chunk) for chunk in chunks] == [FormulaChunk, FormulaChunk, TextChunk]
    assert [chunk.content for chunk in chunks] == ['x', 'y', 'End.']
    assert [chunk.metadata['section'] for chunk in chunks] == ['First', 'First', 'Second']


def test_long_text_respects_chunk_size_around_objects():
    text = 'Sentence. ' * 30
    chunks = RecursiveMarkdownChunker(max_chunk_size=40).split(f'{text}\n\n$$x$$\n\n{text}')
    index = next(i for i, chunk in enumerate(chunks) if isinstance(chunk, FormulaChunk))
    assert index > 1 and len(chunks[index + 1:]) > 1
    assert all(len(chunk.content) <= 40 for chunk in chunks if isinstance(chunk, TextChunk))
    assert [chunk.chunk_order_idx for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.parametrize('path_kind', ['parent', 'absolute', 'symlink'])
async def test_outside_image_is_rejected_before_llm_call(tmp_path, path_kind):
    folder = tmp_path / 'document'
    folder.mkdir()
    outside = tmp_path / 'secret.png'
    outside.write_bytes(b'outside bytes')
    if path_kind == 'parent':
        image_path = '../secret.png'
    elif path_kind == 'absolute':
        image_path = str(outside)
    else:
        (folder / 'link.png').symlink_to(outside)
        image_path = 'link.png'
    llm = AsyncMock(spec=LLM)
    chunker = RecursiveMarkdownChunker(llm=llm)
    with pytest.raises(ValueError, match='document folder'):
        await chunker.a_split(Document(f'![plot]({image_path})', folder=str(folder)))
    llm.batch_chat_completion.assert_not_awaited()


@pytest.mark.parametrize('image_exists', [True, False])
async def test_local_image_attachment_and_missing_image_fallback(tmp_path, image_exists):
    image = tmp_path / 'chart.png'
    payload = b'local image bytes'
    if image_exists:
        image.write_bytes(payload)
    llm = AsyncMock(spec=LLM)
    llm.batch_chat_completion.return_value = ['Chart description']
    chunker = RecursiveMarkdownChunker(llm=llm)
    chunks = await chunker.a_split(Document('![plot](chart.png)', folder=str(tmp_path)))
    request = llm.batch_chat_completion.call_args.args[0][0][-1]['content']
    if image_exists:
        assert request[1]['image_url']['url'] == 'data:image/png;base64,' + base64.b64encode(payload).decode()
    else:
        assert isinstance(request, str)
    assert chunks[0].metadata['description'] == 'Chart description'
