from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# second example below:
hub_llm = HuggingFaceHub(
    repo_id='gpt2',
    model_kwargs={'temperature': 0.7, 'max_length': 100}
)

prompt = PromptTemplate(
    input_variables=["contetnt"],
    template=" You're data scientist,explain {contetnt} to me"
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.run("artificial intelligence")) 
print(hub_chain.run("autoencoders"))
print(hub_chain.run("einstein"))
print(hub_chain.run("deep learning"))