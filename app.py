from io import BytesIO

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

from htmlTemplates import css, bot_template, user_template


CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)


def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        try:
            if hasattr(pdf, "read"):
                pdf_reader = PdfReader(BytesIO(pdf.read()))
            else:
                pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Could not read {getattr(pdf, 'name', 'a file')}: {e}")
    return text


def get_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(raw_text)


def get_vectorstore(chunks):
    if not chunks:
        raise ValueError("No text chunks created from the uploaded PDF.")

    if HuggingFaceEmbeddings is None:
        raise RuntimeError(
            "HuggingFaceEmbeddings not available. Install with: "
            "pip install langchain-huggingface sentence-transformers"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def get_conversationchain(vectorstore):
    llm = ChatOllama(model="llama3.2", temperature=0.2)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory,
    )


def handle_question(question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process a PDF first.")
        return

    try:
        response = st.session_state.conversation.invoke({"question": question})
    except Exception as e:
        st.error(f"Error while answering: {e}")
        return

    st.session_state.chat_history = response.get("chat_history", [])

    for i, msg in enumerate(st.session_state.chat_history):
        text = getattr(msg, "content", str(msg))
        tmpl = user_template if i % 2 == 0 else bot_template
        st.write(tmpl.replace("{{MSG}}", text), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("chat_history", [])

    st.header("Chat with multiple PDFs :books:")
    st.caption("Running locally with Ollama (llama3.2) + HuggingFace embeddings — no API key needed.")

    question = st.text_input("Ask a question about your document:")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs and click 'Process'",
            accept_multiple_files=True,
            type=["pdf"],
        )

        if st.button("Process"):
            if not docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing... (first run downloads the embedding model ~90MB)"):
                raw_text = get_pdf_text(docs)
                if not raw_text.strip():
                    st.error("No readable text found in the uploaded PDF(s). It may be a scanned image PDF.")
                    return

                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversationchain(vectorstore)
                st.success(f"Processed {len(text_chunks)} chunks. Ask away!")


if __name__ == "__main__":
    main()