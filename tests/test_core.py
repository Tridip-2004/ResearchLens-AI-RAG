from langchain_core.documents import Document
from app.ingest import parse_file, chunk_documents
from app.retrieval import rrf
from app.answer import validate_citations, generate_answer
from evaluation.metrics import retrieval_metrics
from evaluation.gate import check

def test_ingest():
    pages = parse_file('demo.txt', b'Rainfall is associated with mosquito breeding. ' * 30)
    chunks = chunk_documents(pages)
    assert chunks and all(c.metadata.get('chunk_id') for c in chunks)

def test_rrf():
    assert rrf([['a', 'b'], ['b', 'a']]) == ['a', 'b']

def test_citations():
    assert validate_citations('Rainfall [1]', 1)
    assert not validate_citations('Rainfall [2]', 1)
    assert not validate_citations('Rainfall', 1)

def test_extractive_answer():
    d = Document(page_content='Rainfall influences mosquito breeding.', metadata={'source': 'demo.txt', 'page': 1, 'chunk_id': 'x'})
    result = generate_answer('What influences mosquito breeding?', [d])
    assert result['citation_valid'] and '[1]' in result['answer']

def test_metrics():
    m = retrieval_metrics(['a', 'b'], {'b'}, k=10)
    assert m['recall_at_10'] == 1 and m['mrr_at_10'] == 0.5

def test_gate_fails_closed():
    assert check({}, min_questions=20)
