import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


class FinTechRAG:

    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 80

    def __init__(self, data_path=None, vector_db_path="./chroma_db"):

        self.data_path = data_path or "./knowledge_base"
        self.vector_db_path = vector_db_path

        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.llm = None

    # ---------- Load Documents ----------

    def chunk_text(self, text):

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - self.CHUNK_OVERLAP

        return chunks

    def load_documents(self):

        loader = DirectoryLoader(
            self.data_path,
            glob="*.pdf",
            loader_cls=PyMuPDFLoader
        )

        final_docs = []

        for doc in loader.load():

            chunks = self.chunk_text(doc.page_content)

            for chunk in chunks:
                final_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    )
                )

        return final_docs

    # ---------- Vector Store ----------

    def create_vector_store(self):

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(self.vector_db_path):

            print("Loading existing vector DB")

            self.vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=embedding
            )

        else:

            print("Creating new vector DB")

            docs = self.load_documents()

            self.vectorstore = Chroma.from_documents(
                docs,
                embedding,
                persist_directory=self.vector_db_path
            )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    # ---------- LLM ----------

    def setup_llm(self):

        print("Connecting to Gemini...")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )

        print("Gemini Ready")


    # ---------- QA Chain ----------
    def setup_chain(self):

        prompt = PromptTemplate(
            template="""
    You are a financial assistant.

    Answer ONLY using the provided context.
    If answer is missing, say:
    "I don't know based on provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    # ---------- Initialize ----------

    def initialize(self):

        print("Initializing RAG")

        self.create_vector_store()
        self.setup_llm()
        self.setup_chain()

    # ---------- Ask ----------

    def ask(self, query) -> Dict[str, Any]:

        result = self.qa_chain.invoke({"query": query})

        print("DEBUG RESULT:", result)

        sources = []

        for doc in result["source_documents"]:
            sources.append({
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:200]
            })

        return {
            "answer": result["result"],
            "sources": sources,
            "query": query
        }
