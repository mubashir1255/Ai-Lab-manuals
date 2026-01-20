# Install required libraries
!pip install langchain langchain-community pypdf pandas bs4 requests

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader
)

# ----------- Load PDF -----------
pdf_loader = PyPDFLoader("lecture_notes.pdf")
pdf_docs = pdf_loader.load()

# ----------- Load Web Page -----------
web_url = "https://en.wikipedia.org/wiki/Reinforcement_learning"
web_loader = WebBaseLoader(web_url)
web_docs = web_loader.load()

# ----------- Load CSV -----------
csv_loader = CSVLoader("students.csv")
csv_docs = csv_loader.load()

# ----------- Comparison Output -----------

print("===== PDF LOADER =====")
print("Content Format (First 300 chars):")
print(pdf_docs[0].page_content[:300])
print("\nMetadata:")
print(pdf_docs[0].metadata)

print("\n==============================\n")

print("===== WEB LOADER =====")
print("Content Format (First 300 chars):")
print(web_docs[0].page_content[:300])
print("\nMetadata:")
print(web_docs[0].metadata)

print("\n==============================\n")

print("===== CSV LOADER =====")
print("Content Format (First 300 chars):")
print(csv_docs[0].page_content[:300])
print("\nMetadata:")
print(csv_docs[0].metadata)
