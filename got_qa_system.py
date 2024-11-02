from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from loader import load_pdf_text, split_text
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class GoTQASystem:
    def __init__(self, pdf_path, api_token):
        self.pdf_path = pdf_path
        self.api_token = api_token
        self.setup_system()

    def setup_system(self):
        text = load_pdf_text(self.pdf_path)
        chunks = split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        self.llm = HuggingFaceHub(api_token=self.api_token, model="gpt-neo-1.3B")
        self.qa_chain = load_qa_chain(self.llm, chain_type="stuff")

    def answer_question(self, question):
        docs = self.vector_store.similarity_search(question, k=3)
        return self.qa_chain.run(input_documents=docs, question=question)
