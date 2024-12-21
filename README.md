# LangChain-Practice
Starte with my LangChain Journet with Ollama

LangChain is an open-source framework designed to build applications powered by large language models (LLMs). It simplifies the entire lifecycle of developing, deploying, and managing LLM applications. LangChain help us create LLM powered apps

>>pip install langchain

OpenAI is LLM model
we create openai api and invoke this through langchain

**Creating my own Environment for project**
>>python -m venv environment_name
>>.venv\Scripts\activate
>>pip install -r requirements.txt


**Ollama.exe contains**
server and client (ollama run phi((CLI)
when we run the command "ollama run phi" it does api request to server

1. Install Ollama.exe file
then
2. CMD
>>Ollama (to check if its there)
>>Where Ollama (to check if it insalled locaton)
>> ollama pull ollama_model_name (example: phi)

3. python ex:
Ollama(model="phi")
