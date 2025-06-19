from handlers.rag_handler import ChromaRAGHandler
from utils.parser import parse_chroma_metadata

rag_handler = ChromaRAGHandler()

query = "What is the name of the main character in the story?"
user_id = "1"


def test_query_vector_database():
    status, answer = rag_handler.query_vector_database(
        {"title": "Story.txt"},
        query,
        user_id,
    )
    if status:
        print(answer)
    else:
        print(answer)


def test_delete_file_with_name_from_vector_database():
    status, answer = rag_handler.delete_file_with_name_from_vector_database(
        "Garrett Davis.pdf",
        user_id,
    )
    if status:
        print(answer)
    else:
        print(answer)


def test_delete_collection_from_vector_database():
    status, answer = rag_handler.delete_collection_from_vector_database(
        user_id,
    )
    if status:
        print(answer)
    else:
        print(answer)


def test_save_to_web_collection():
    status, answer = rag_handler.save_to_web_collection(
        "world",
        {
            "user_id": "1",
            "session_id": "3",
        },
    )
    if status:
        print(answer)
    else:
        print(answer)


if __name__ == "__main__":
    # test_query_vector_database()
    # test_delete_file_with_name_from_vector_database()
    # test_delete_collection_from_vector_database()
    metadata = {"user_id": "1", "session_id": "1"}
