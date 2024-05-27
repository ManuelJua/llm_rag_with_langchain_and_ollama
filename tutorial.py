from langchain_community.llms import Ollama
from datetime import datetime
llm=Ollama(base_url="http://localhost:11434",model="phi3")


from langchain_community.document_loaders import PyMuPDFLoader
import os
data_pdf=[]
for document in os.listdir(os.path.abspath(".")):
    if document[-4:]==".pdf":
        loader = PyMuPDFLoader(os.path.abspath(f"./{document}"))
        data_pdf.extend(loader.load())


        # data_pdf.extend(loader.load())


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
docs = text_splitter.split_documents(data_pdf)
print(f"length docs:{len(docs)} length pdf:{len(data_pdf)}")


from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from langchain_community.vectorstores import Chroma

vs = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory="chroma_db_dir",  # Local mode with in-memory storage only
    collection_name="papers_interfaces"
)

vectorstore = Chroma(embedding_function=embed_model,
                     persist_directory="chroma_db_dir",
                     collection_name="papers_interfaces")
retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

from langchain.prompts import PromptTemplate

custom_prompt_template = """ Please, answer the question using only the information provided:

context: {context}
question: {question}
If you cannot find and answer, please say you don not know the answer.
Answer:
"""
prompt = PromptTemplate(template=custom_prompt_template,
                        input_variables=['context', 'question'])


from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs={"prompt": prompt})

query=input('\nPlease ask a question\n')

while query!='salir':
    start=datetime.now()
    response = qa.invoke({"query": query})
    end=datetime.now()
    print("\n",response['result'],"\n",f"Response time: {end-start}\n")
    query=input('\nPlease ask a question\n')