import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

from config import RAG_SYSTEM_TEMPLATE
from utils import error_handler


class ChromaRAGHandler:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    @error_handler
    def save_to_vector_database(self, text: str, metadata: dict, user_id: str) -> str:
        docs = self.text_splitter.create_documents([text])
        documents = []
        for doc in docs:
            documents.append(Document(page_content=doc.page_content, metadata=metadata))

        collection_name = f"collection_{user_id}"
        Chroma.from_documents(
            collection_name=collection_name,
            documents=documents,
            embedding=self.embeddings_model,
            persist_directory="chroma_db",
        )
        return "Successfully saved to vector database"

    @error_handler
    def query_vector_database(self, query: str, user_id: str) -> str:
        collection_name = f"collection_{user_id}"
        db = Chroma(
            collection_name=collection_name,
            persist_directory="chroma_db",
            embedding_function=self.embeddings_model,
        )

        # TODO: need to add filter to the query
        # retriever = db.as_retriever(k=10, filter={"title": "Garrett Davis.pdf"})
        retriever = db.as_retriever(k=10)
        docs = retriever.invoke(query)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    RAG_SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        document_chain = create_stuff_documents_chain(
            self.llm, question_answering_prompt
        )
        answer = document_chain.invoke(
            {
                "context": docs,
                "messages": [HumanMessage(content=query)],
            }
        )
        return answer

        # Streaming feature
        # query_transform_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         MessagesPlaceholder(variable_name="messages"),
        #         (
        #             "user",
        #             "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        #         ),
        #     ]
        # )
        # query_transforming_retriever_chain = RunnableBranch(
        #     (
        #         lambda x: len(x.get("messages", [])) == 1,
        #         # If only one message, then we just pass that message's content to retriever
        #         (lambda x: x["messages"][-1].content) | retriever,
        #     ),
        #     # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
        #     query_transform_prompt | self.llm | StrOutputParser() | retriever,
        # ).with_config(run_name="chat_retriever_chain")
        # SYSTEM_TEMPLATE = """
        # Answer the user's questions based on the below context.
        # If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

        # <context>
        # {context}
        # </context>
        # """

        # question_answering_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             SYSTEM_TEMPLATE,
        #         ),
        #         MessagesPlaceholder(variable_name="messages"),
        #     ]
        # )

        # document_chain = create_stuff_documents_chain(
        #     self.llm, question_answering_prompt
        # )
        # conversational_retrieval_chain = RunnablePassthrough.assign(
        #     context=query_transforming_retriever_chain,
        # ).assign(
        #     answer=document_chain,
        # )
        # result = conversational_retrieval_chain.invoke(
        #     {
        #         "messages": [
        #             HumanMessage(content=query),
        #         ],
        #     }
        # )
        # print(result["answer"])
        # stream = conversational_retrieval_chain.stream(
        #     {
        #         "messages": [
        #             HumanMessage(content=query),
        #         ],
        #     }
        # )

        # for chunk in stream:
        #     print(chunk)

    @error_handler
    def delete_collection_from_vector_database(self, user_id: str) -> str:
        """
        Delete a collection from the vector database
        Args:
            user_id: str
        Return:
            A success message
        """

        collection_name = f"collection_{user_id}"
        db = Chroma(
            collection_name=collection_name,
            persist_directory="chroma_db",
            embedding_function=self.embeddings_model,
        )

        # FIXEME: if Chroma removes embeddings while removing collection
        # delete all embeddings first, then remove the collection
        collection = db.get()
        db.delete(ids=collection["ids"])
        db.delete_collection()
        return "Successfully deleted collection from vector database"

    @error_handler
    def delete_file_by_name_from_vector_database(
        self, file_name: str, user_id: str
    ) -> str:
        """
        Delete a file from the vector database by its name
        Args:
            file_name: str
            user_id: str
        Return:
            A success message
        """

        collection_name = f"collection_{user_id}"
        db = Chroma(
            collection_name=collection_name,
            persist_directory="chroma_db",
            embedding_function=self.embeddings_model,
        )
        collection = db.get(where={"title": {"$eq": file_name}})
        db.delete(ids=collection["ids"])

        return "Successfully deleted from vector database"
