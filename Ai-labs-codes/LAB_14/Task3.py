# Step 1: Install required libraries
!pip install langchain langchain-community pypdf

# Step 2: Upload PDF file
from google.colab import files
uploaded = files.upload()

# Step 3: Load PDF using LangChain
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("lecture_notes.pdf")
documents = loader.load()

# Step 4: Count total pages
print("Total number of pages:", len(documents))

# Step 5: Display content of the first page
print("\nContent of First Page:\n")
print(documents[0].page_content)
