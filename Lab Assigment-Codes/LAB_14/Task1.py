!pip install langchain langchain-community langchain-core openai python-dotenv
import langchain
print(langchain.__version__)
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Initialize the model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

# Create a LangChain
chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# Run the chain
response = chain.run("Reinforcement Learning")
print(response)
