from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

import os

import streamlit as st

load_dotenv()

st.set_page_config(page_title="FinTech Customer Support Assistant")
st.title("FinTech Customer Support Assistant")
st.caption("Ask Financial Questions")


# build the retriever once per session
if "retriever" not in st.session_state:
    with st.spinner("Loading documents..."):
        docs = DirectoryLoader("Dataset", glob="**/*.pdf", loader_cls = PyPDFLoader).load()

        # preprocessing: fix the rupee sign and remove bold markers
        for d in docs:
            d.page_content = d.page_content.replace("■", "₹").replace("**", "")

        # get each PDF's title (first line of its first page) and add it to every page
        titles = {}
        for d in docs:
            if d.metadata["page"] == 0:
                titles[d.metadata["source"]] = d.page_content.strip().split("\n")[0]
        for d in docs:
            d.metadata["title"] = titles.get(d.metadata["source"], os.path.basename(d.metadata["source"]))

        # chunking
        chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100).split_documents(docs)

        # embeddings + vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

NOT_FOUND = "I could not find this information in the knowledge base."

prompt = ChatPromptTemplate.from_template(
    """You are a helpful FinTech customer support assistant.
Answer the question using only the context below.
If the answer is not in the context, reply exactly with: {not_found}
Otherwise, after the answer, add a new line in this format:
Source: <document title> — Section <number>
Take the document title from the "Document title" line and the section number from the context. If the context has no section number, write only the document title.

Chat history:
{history}

Context:
{context}

Question: {question}

Answer:"""
)

chain = prompt | llm | StrOutputParser()


if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

question = st.chat_input("Type your question here...")

if question:
    st.chat_message("user").write(question)

    results = st.session_state.retriever.invoke(question)
    context = "\n\n".join("Document title: " + r.metadata["title"] + "\n" + r.page_content for r in results)
    history = "\n".join(role + ": " + text for role, text in st.session_state.history[-6:])

    answer = chain.invoke({"not_found": NOT_FOUND, "history": history, "context": context, "question": question})

    st.chat_message("assistant").write(answer)

    st.session_state.history.append(("user", question))
    st.session_state.history.append(("assistant", answer))