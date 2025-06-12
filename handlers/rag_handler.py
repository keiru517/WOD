import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class ChromaRAGHandler:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    def save_to_vector_database(self, text: str, metadata: dict) -> str:
        docs = self.text_splitter.create_documents([text])
        documents = []
        for doc in docs:
            documents.append(Document(page_content=doc.page_content, metadata=metadata))
        # embeddings = self.embeddings_model.embed_documents(texts)
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings_model,
            persist_directory="chroma_db",
        )
        # db = Chroma(
        #     persist_directory="chroma_db",
        #     embedding_function=self.embeddings_model,
        # )
        query = "I would like to get the list of companies Garrett Davis has worked at"
        retriever = db.as_retriever(k=10, filter={"title": "Garrett Davis.pdf"})
        docs = retriever.invoke(query)
        SYSTEM_TEMPLATE = """
        Answer the user's questions based on the below context. 
        If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

        <context>
        {context}
        </context>
        """
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_TEMPLATE,
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
        print(answer)
