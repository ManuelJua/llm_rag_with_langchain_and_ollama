# llm_rag_with_langchain_and_ollama

This repository contains a Python script that integrates various components from the langchain library to perform document retrieval and question-answering tasks using **large language models** and **Retrieval Augmented Generation (RAG)**, with the support of Ollama for interacting with these models and Docker for containerization.

## Overview

The script performs the following functions:
- Loads data from PDF documents.
- Splits text into manageable chunks.
- Embeds the text for vector representation.
- Creates a vector store for efficient document retrieval.
- Sets up a retrieval-based question-answering system using Ollama to interact with large language models and RAG.

## Requirements

- langchain-community
- langchain
- langchain-chroma
- langchain-core
- langchain-openai
- langchain-text-splitters
- fastembed
- pymupdf