# Install required libraries
!pip install langchain langchain-community pandas

# Import CSV Loader
from langchain_community.document_loaders import CSVLoader

# Load the CSV file
# (File already uploaded in Colab environment)
loader = CSVLoader(file_path="/mnt/data/students (1).csv")
documents = loader.load()

# Inspect how many documents were created (each row = one document)
print("Total number of documents created:", len(documents))

# Print one document sample
print("\nSample Document:\n")
print("Page Content:")
print(documents[0].page_content)

print("\nMetadata:")
print(documents[0].metadata)
