import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from utils import load_and_split_pdf, tag_scenes_and_characters, GoTQASystem

# Specify the path to your Game of Thrones PDF
pdf_path = "Game of Thrones.pdf"
split_books = load_and_split_pdf(pdf_path)
print(f"Loaded and split {len(split_books)} sections from PDF.")

# Initialize and tag documents
tagged_documents = tag_scenes_and_characters(split_books)

# Create and setup Q&A system
huggingface_token = "your_huggingface_token_here"
qa_system = GoTQASystem(huggingface_token)
qa_system.create_embeddings(tagged_documents)
qa_system.setup_conversation_chain()
