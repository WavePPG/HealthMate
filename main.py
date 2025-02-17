import requests
from io import BytesIO
from PIL import Image
import uvicorn
import numpy as np
import os
import faiss
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, FlexSendMessage,
    BubbleContainer, CarouselContainer, BoxComponent,
    TextComponent, ButtonComponent, URIAction
)
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
from typing import Dict
from contextlib import asynccontextmanager

ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "FfVoDvJvHY3kYAqiA/Rgr32EBpgyDfssfV5aX5L+8Zry5vf1yyc9qRcqkRAru52gJzYQJlgd4jKZIFoMo/iQlLPRsz+S6NO12SrIYFn2UzCV/iOv7wIdJGnVVgNcn+rem7ej+0FGKOdzQX4/VYGZuwdB04t89/1O/w1cDnyilFU=")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "99a40583e525a2daf0494e3198c45907")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCR8-cNrPtbFs9rTVl49BBGcGHBw8O3SRY")

# Setup LINE API
line_bot_api = LineBotApi(ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

class RAGSystem:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.create_faiss_index()

    def add_document(self, text: str, metadata: dict = None):
        embedding = self.embedding_model.encode([text])[0]
        self.database['documents'].append(text)
        self.database['embeddings'].append(embedding.tolist())
        self.database['metadata'].append(metadata or {})
        self.create_faiss_index()

    def create_faiss_index(self):
        if not self.database['embeddings']:
            self.index = None
            return
        embeddings = np.array(self.database['embeddings'], dtype='float32')
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve_documents(self, query: str, top_k: int = 3):
        if not self.database['embeddings'] or self.index is None:
            return []
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        D, I = self.index.search(query_embedding, top_k)
        return [self.database['documents'][i] for i in I[0] if i < len(self.database['documents'])]

    def clear_database(self):
        self.database = {
            'documents': [],
            'embeddings': [],
            'metadata': []
        }
        self.index = None

rag = RAGSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Add your sample documents here
    sample_documents = [EMERGENCY_MANUAL, WATCH_ELEPHANT_MANUAL, CHECK_ELEPHANT_MANUAL, OFFICER_MANUAL]
    for doc in sample_documents:
        rag.add_document(doc)
    yield
    rag.clear_database()

app = FastAPI(lifespan=lifespan)

EMERGENCY_MANUAL = """คู่มือการใช้งานฟีเจอร์ "Emergency" 🆘
ฟังก์ชันหลัก:
คำแนะนำฉุกเฉิน: กดปุ่ม "Emergency" เพื่อรับคำแนะนำเมื่อเกิดเหตุการณ์ฉุกเฉินต่างๆ
ถาม-ตอบกับบอท: พิมพ์คำถามเกี่ยวกับสถานการณ์ฉุกเฉิน เช่น "ช้างเหยียบรถต้องทำอย่างไร" เพื่อรับคำตอบในทันที
"""
WATCH_ELEPHANT_MANUAL = """เมื่อช้างเข้าใกล้ในสถานการณ์ฉุกเฉิน ให้ปฏิบัติตามขั้นตอนต่อไปนี้:
1.ตั้งสติให้มั่น: พยายามสงบสติอารมณ์ อย่าแสดงอาการตื่นตระหนกหรือหวาดกลัว
2.หลีกเลี่ยงสายตา: อย่าสบตากับช้างโดยตรง ให้มองลงพื้นหรือมองทางอื่น
3.ถอยห่างอย่างช้าๆ: ค่อยๆ ถอยหลังออกห่างจากช้าง อย่าทำการเคลื่อนไหวกะทันหัน
4.มองหาที่กำบัง: พยายามเข้าไปอยู่ในที่ที่มีสิ่งกีดขวาง เช่น หลังต้นไม้ใหญ่หรือกำแพง
5.แจ้งเจ้าหน้าที่: รีบติดต่อขอความช่วยเหลือจากเจ้าหน้าที่ทันที โทร 086-092-6529 ซึ่งเป็นเบอร์ติดต่อ ศูนย์บริการนักท่องเที่ยว
"""
CHECK_ELEPHANT_MANUAL = """🐘 ตรวจสอบช้างก่อนออกเดินทาง! เช็คความปลอดภัยจากช้างก่อนเที่ยวได้ที่นี่ 👉 คลิกเลย
"""
OFFICER_MANUAL = """📞 ติดต่อเจ้าหน้าที่
เหตุฉุกเฉินทุกกรณี: โทร 1669 (บริการตลอด 24 ชั่วโมง)
ศูนย์บริการนักท่องเที่ยว: โทร 086-092-6529
ที่ทำการอุทยานแห่งชาติเขาใหญ่: โทร 086-092-6527
"""


def get_manual_response(user_message: str) -> str:
    user_message = user_message.strip().lower()
    manuals = {
        "emergency": EMERGENCY_MANUAL,
        "คู่มือการใช้งาน": EMERGENCY_MANUAL,
        "emergency เกิดเหตุฉุกเฉินทำยังไง": WATCH_ELEPHANT_MANUAL,
        "มีเหตุร้ายใกล้ตัว": WATCH_ELEPHANT_MANUAL,
        "ตรวจสอบช้างก่อนเดินทาง": CHECK_ELEPHANT_MANUAL,
        "ติดต่อเจ้าหน้าที่": OFFICER_MANUAL,
        "contact officer": OFFICER_MANUAL
    }
    return manuals.get(user_message)

def create_bubble_container(text: str) -> BubbleContainer:
    return BubbleContainer(
        header=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text="WildSafe",
                    weight='bold',
                    align='center',
                    color='#FFFFFF',
                    size='xl'
                )
            ],
            background_color='#27AE60'
        ),
        body=BoxComponent(
            layout='vertical',
            contents=[
                TextComponent(
                    text=text,
                    wrap=True,
                    size='sm'
                )
            ]
        ),
        footer=BoxComponent(
            layout='vertical',
            contents=[
                ButtonComponent(
                    style='primary',
                    action=URIAction(
                        label='GO MAP',
                        uri='https://aprlabtop.com/Honey_test/chang_v23.php'
                    )
                )
            ]
        )
    )

def create_flex_message(text: str) -> FlexSendMessage:
    bubble = create_bubble_container(text)
    return FlexSendMessage(alt_text="WildSafe Message", contents=bubble)

def create_carousel_message(texts: list) -> FlexSendMessage:
    bubbles = [create_bubble_container(text) for text in texts]
    carousel = CarouselContainer(contents=bubbles)
    return FlexSendMessage(alt_text="WildSafe Carousel", contents=carousel)

@app.post('/message')
async def message(request: Request):
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(status_code=400, detail="X-Line-Signature header is missing")
    
    body = await request.body()
    try:
        handler.handle(body.decode("UTF-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return {"status": "ok"}

@handler.add(MessageEvent, message=(TextMessage, ImageMessage))
def handle_message(event: MessageEvent):
    if isinstance(event.message, TextMessage):
        user_message = event.message.text
        manual_response = get_manual_response(user_message)
        
        if manual_response:
            reply = create_flex_message(manual_response)
        else:
            # Check if the message matches predefined keywords exactly
            relevant_to_rag = any(user_message.strip().lower() == phrase for phrase in ['ฉุกเฉิน', 'ช้าง', 'เจ้าหน้าที่'])
            
            if relevant_to_rag:
                retrieved_docs = rag.retrieve_documents(user_message, top_k=3)
                if retrieved_docs:
                    texts = ["ดูข้อมูลเพิ่มเติมที่นี่" if "http" in doc else doc for doc in retrieved_docs]
                    reply = create_carousel_message(texts)
                else:
                    # Use Gemini if RAG has no results
                    gemini_response = model.generate_content(user_message + " ให้สรุปสั้นๆใน 2-3 บรรทัด")
                    reply = create_flex_message(gemini_response.text.strip().split("\n")[:3])
            else:
                # Use Gemini for non-matching queries
                gemini_response = model.generate_content(user_message + " ให้สรุปสั้นๆใน 2-3 บรรทัด โดยเกี่ยวกับสุขภาพ")
                reply = create_flex_message("\n".join(gemini_response.text.strip().split("\n")[:3]))

        line_bot_api.reply_message(
            event.reply_token,
            [reply]
        )

    elif isinstance(event.message, ImageMessage):
        try:
            headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
            url = f"https://api-data.line.me/v2/bot/message/{event.message.id}/content"
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
            
            if image.size[0] * image.size[1] > 1024 * 1024:
                message = "ขอโทษครับ ภาพมีขนาดใหญ่เกินไป กรุณาลดขนาดภาพและลองใหม่อีกครั้ง"
            else:
                try:
                    gemini_response = model.generate_content("อธิบายรูปภาพนี้ ให้สรุปสั้นๆใน 2-3 บรรทัด")
                    message = "\n".join(gemini_response.text.strip().split("\n")[:3])
                except Exception:
                    message = "ขณะนี้ระบบไม่สามารถประมวลผลรูปภาพได้ กรุณาสอบถามด้วยข้อความแทนค่ะ 🙏🏻"
                
        except Exception:
            message = "เกิดข้อผิดพลาด, กรุณาลองใหม่อีกครั้ง🙏🏻"
            
        reply = create_flex_message(message)
        line_bot_api.reply_message(event.reply_token, [reply])



if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")

