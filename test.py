from handlers.rag_handler import ChromaRAGHandler

rag_handler = ChromaRAGHandler()
query = "Give me a summary about Garrett Davis's career"
user_id = "1"
status, answer = rag_handler.query_vector_database(
    query,
    user_id,
)
if status:
    print(answer)
else:
    print(answer)
