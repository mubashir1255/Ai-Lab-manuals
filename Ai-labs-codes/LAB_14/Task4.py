# Install required libraries
!pip install langchain langchain-community bs4 requests

# Import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

# Webpage URL to load
url = "https://en.wikipedia.org/wiki/Reinforcement_learning"

# Load webpage
loader = WebBaseLoader(url)
documents = loader.load()

# Display extracted text (first 1000 characters for clarity)
print("Extracted Text Content (Preview):\n")
print(documents[0].page_content[:1000])

# Display metadata
print("\nMetadata Information:\n")
print(documents[0].metadata)
