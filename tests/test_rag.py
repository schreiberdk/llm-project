from ml_logic.context_retrieval import retrieve_context, load_vectorstore

def test_context_retrieval():
    vs = load_vectorstore("vector_db")
    query = "medical symptoms of fever"
    context = retrieve_context(query, vs)
    assert len(context) > 0  # Ensure it returns some text
