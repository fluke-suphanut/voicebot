# Thai Voice Q&A Chatbot with LangChain & Ollama

โปรเจคนี้คือ **Voice-based Chatbot ภาษาไทย** ที่ให้ผู้ใช้พูดคำถาม และระบบจะ

1. แปลงเสียงเป็นข้อความ (Speech-to-Text)
2. ค้นหาความรู้จากไฟล์ข้อความ (`text02new.txt`) ด้วย **LangChain + VectorDB**
3. ใช้ **Ollama LLM** (LLaMA 3.1) สร้างคำตอบ
4. แปลงคำตอบเป็นเสียงภาษาไทยกลับมา (Text-to-Speech)

## Features

* รับเสียงพูดภาษาไทย → แปลงเป็นข้อความ (Google Speech Recognition)
* ค้นหาคำตอบจาก knowledge base (`text02new.txt`)
* ใช้ **HuggingFace Embeddings (LaBSE)** สำหรับภาษาไทย
* ใช้ **Ollama LLM** (LLaMA 3.1 8B) เป็นตัวตอบคำถาม
* อ่านคำตอบออกเสียงภาษาไทยด้วย **gTTS + Pygame**
* ควบคุมด้วยเสียง (พูด `"exit"` เพื่อออกจากโปรแกรม)

## Requirements

ติดตั้ง dependencies ด้วย `pip`:

```bash
pip install speechrecognition pygame gTTS langchain langchain-community langchain-ollama langchain-huggingface chromadb
```

นอกจากนี้ต้องมี:

* **Ollama** ติดตั้งในเครื่อง และโหลดโมเดล `llama3.1:8b`
* ไมโครโฟนสำหรับ input

## Running

รันสคริปต์:

```bash
python voicebot.py
```

โปรแกรมจะ:

1. ปรับระดับเสียง
2. รอให้พูดคำถาม
3. แสดงผลลัพธ์ใน console และตอบกลับด้วยเสียง

พูด `"exit"` เพื่อจบการทำงาน

## Example

**User (พูด):**

```
ภาควิชานี้ก่อตั้งเมื่อไหร่
```

**Bot (เสียงตอบกลับ):**

```
ภาควิชาวิศวกรรมไฟฟ้า ก่อตั้งตั้งแต่ปี พ.ศ. 2550
```

## Note

* แนะนำให้ใส่ค่า `channel_secret`, `channel_access_token` และ config อื่น ๆ ใน `.env` ไฟล์ (ถ้ามีการเชื่อมต่อภายนอก)
* การใช้ Google Speech Recognition ต้องต่ออินเทอร์เน็ต

