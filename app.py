import json
import os
import openai
import chainlit as cl
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
 
@cl.on_chat_start
def start_chat():
    # OpenAI init
    model_name = "gpt-4"
    openai_api_embeddings_deployment_name = os.getenv("OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME")
    openai.api_type = "azure"  
    openai.api_version = os.getenv("OPENAI_API_VERSION")  
    openai.api_key = os.getenv("CHATGPT_OPENAI_API_KEY")  
    openai.api_base = os.getenv("CHATGPT_OPENAI_API_BASE")  
    print(f"openai.api_base={openai.api_base}")

    # Azure search init
    searchservice = os.getenv("AZURE_SEARCH_SERVICE")
    azure_search_endpoint = f"https://{searchservice}.search.windows.net/"
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")  
    azure_search_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  

    embeddings = AzureOpenAIEmbeddings(deployment=openai_api_embeddings_deployment_name, chunk_size=1, openai_api_type=openai.api_type, azure_endpoint= openai.api_base, openai_api_key= openai.api_key) 
    embedding_function=embeddings.embed_query

    # Connect to Azure Cognitive Search
    acs = AzureSearch(azure_search_endpoint,
                    azure_search_key,
                    index_name,
                    embedding_function=embedding_function,
                    search_type="similarity")

    model = AzureChatOpenAI(deployment_name=model_name, azure_endpoint= openai.api_base, openai_api_key=openai.api_key, temperature=0)

    retriever=acs.as_retriever()

    template = """
    Assistant helps the company employees with their questions on company policies, roles. 
    Always include the source metadata for each fact you use in the response. Use square brakets to reference the source, e.g. [role_library_pdf-10]. 
    Properly format the output for human readability with new lines.
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # memory = ConversationBufferMemory()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # conv_chain  = ConversationChain(llm=chain, memory=memory)

    cl.user_session.set("runnable", chain)
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": template}],
    )


@cl.on_message
async def main(message: cl.Message):
    # message_history = cl.user_session.get("message_history")
    # print(message_history)
    # message_history.append({"role": "user", "content": message.content})
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
         message.content,
         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], configurable={"session_id": "foobar"}),
    ):
        await msg.stream_token(chunk)

    # message_history.append({"role": "assistant", "content": msg.content})
    await msg.send()
