# Using chainlit and langchain with Azure Cognitive Search Vector Search 

This project demonstrates how to use chainlit, langchain and OpenAI with your own data. 
The goal is to show how to use these technologies together to build a simple web app that can allows converse with ChtGPT about your own data.

## Prerequisites

Before you begin, you'll need to have the following:

- An Azure account with access to Azure Cognitive Search and data loaded into it. For example you can use the [azure-openai-vector-search_langchain](https://github.com/romicgd/azure-openai-vector-search_langchain) to load data into Azure Cognitive Search.
- A dataset of documents that you want to search through
- An OpenAI API key

.env needs to contain following 
```
   AZURE_SEARCH_SERVICE=...
   AZURE_SEARCH_INDEX_NAME=...
   AZURE_SEARCH_ADMIN_KEY=...
   CHATGPT_OPENAI_API_BASE=https://%%%%%.openai.azure.com/
   CHATGPT_OPENAI_API_KEY=...
   OPENAI_API_VERSION=2023-05-15
   AZURESEARCH_FIELDS_CONTENT_VECTOR=contentVector
   OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME=...
```