from handlers.rag_handler import ChromaRAGHandler

rag_handler = ChromaRAGHandler()

query = "Give me a summary about Garrett Davis's career"
user_id = "1"


def test_save_to_vector_database():
    status, answer = rag_handler.query_vector_database(
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


if __name__ == "__main__":
    # test_save_to_vector_database()
    test_delete_file_with_name_from_vector_database()
