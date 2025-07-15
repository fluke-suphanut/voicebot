import warnings
warnings.filterwarnings("ignore")

import sys
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import speech_recognition as sr
import io
import pygame
from gtts import gTTS
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

r = sr.Recognizer()
pygame.mixer.init()

def tts(text_out,lang):
    with io.BytesIO() as f:
        speak = gTTS(text=text_out,lang=lang)
        speak.save("temp_speak.mp3")
        pygame.mixer.music.load("temp_speak.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


if _name_ == "_main_":
    # 1) โหลดไฟล์ข้อความภาษาไทย (หรือเอกสารใด ๆ)
    text_file_path = "text02new.txt"
    loader = TextLoader(text_file_path, encoding="utf-8")
    data = loader.load()

    # 2) สร้าง Text Splits
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    # 3) เลือกโมเดล Embeddings ภาษาไทย 
    embedding_model_name = "sentence-transformers/LaBSE"

    thai_embedding = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={
            "device": "cpu"  # เลือก "cuda" หากต้องการใช้ GPU
        }
    )

    # 4) สร้าง VectorStore โดยใช้ Embeddings ภาษาไทย
    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=thai_embedding)

    # 5) สร้าง loop สำหรับถาม-ตอบ
    with sr.Microphone() as source:
        print("กำลังปรับระดับเสียง... โปรดรอสักครู่")
        r.adjust_for_ambient_noise(source, duration=3)
        
        while True:
            print("\nกรุณาพูด...")
            audio = r.listen(source)
            try:
                my_text = r.recognize_google(audio,language="th")
                print("คุณพูดว่า: " + my_text)


                query = my_text
                if query.lower() == "exit":
                    break
                if query.strip() == "":
                    continue

                # Prompt Template
                template = """Use the following pieces of context to answer the question at the end.
        These are your information. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.
        {context}
        Question: {question}
        Helpful Answer:"""
                QA_CHAIN_PROMPT = PromptTemplate(
                    input_variables=["context", "question"],
                    template=template,
                )

                # สร้าง LLM (ใช้ Ollama ตามตัวอย่าง)
                llm = OllamaLLM(
                    model="llama3.1:8b",
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )

                # สร้าง RetrievalQA
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                )

                # 6) ยิงคำถามและรับคำตอบ
                result = qa_chain.invoke({"query": query})
                tts(result['result'],"th")
            except:
                pass