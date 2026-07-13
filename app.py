from io import BytesIO

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain.memory.buffer import ConversationBufferMemory
except ModuleNotFoundError:
    from langchain.memory import ConversationBufferMemory

try:
    from langchain.chains import ConversationalRetrievalChain
except ModuleNotFoundError:
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

from htmlTemplates import css, bot_template, user_template

custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        if hasattr(pdf, "read"):
            file_bytes = pdf.read()
            pdf_reader = PdfReader(BytesIO(file_bytes))
        else:
            pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(chunks):
    if not chunks:
        raise ValueError("No text chunks created from the uploaded PDF.")

    if HuggingFaceEmbeddings is not None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        except Exception as e:
            st.warning(f"HuggingFaceEmbeddings failed: {e}. Falling back to OpenAIEmbeddings.")
            embeddings = OpenAIEmbeddings()
    else:
        embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def get_conversationchain(vectorstore):
    llm = ChatOpenAI(temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
    )
    return conversation_chain

def handle_question(question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return

    response = st.session_state.conversation.invoke({"question": question})
    st.session_state.chat_history = response.get("chat_history", [])

    for i, msg in enumerate(st.session_state.chat_history):
        text = getattr(msg, "content", str(msg))
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", text), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", text), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    question = st.text_input("Ask question from your document:")

    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'",
            accept_multiple_files=True,
        )

        if st.button("Process"):
            if not docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(docs)
                if not raw_text.strip():
                    st.error("No readable text found in the uploaded PDF(s).")
                    return

                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversationchain(vectorstore)
                st.success("PDF processed successfully.")

if __name__ == "__main__":
    main()