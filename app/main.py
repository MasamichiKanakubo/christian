import os
import json
import re
import requests
import logging
import asyncio
import aiohttp
from openai import OpenAI
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextSendMessage,
    QuickReply,
    QuickReplyButton,
    MessageAction,
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging


# from app.repositories.scrapbox_repository import ScrapboxRepository

load_dotenv()

# scrapbox_repository = ScrapboxRepository(os.getenv("SCRAPBOX_PROJECT_NAME"))

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

url = "https://scrapbox.io/api/pages/christian-beginners/"

question_re_pattern = re.compile(r"\?")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/")
async def callback(request: Request):
    body = await request.body()
    data_json = json.loads(body)

    if data_json["events"]:
        try:
            await handle_message(data_json)
        except IndexError:
            return {"error": "Invalid event data"}
    return {"error": "Events not found"}


async def handle_message(data_json):
    incoming_text = data_json["events"][0]["message"]["text"]
    reply_token = data_json["events"][0]["replyToken"]

    if incoming_text == "お問い合わせできるところどこ？":
        return
    

    if incoming_text == "オタ姫ちゃん何言ってるかわからないよ...":
        not_text = "そっか😢、力になれなくてごめんね🙏次は答えてみせるから...💪💖"
        yet_answer_with_quick_reply = TextSendMessage(
            text=not_text,
            quick_reply=QuickReply(
                items=[
                    QuickReplyButton(
                        action=MessageAction(
                            label="もう一度質問する", text="もう一回質問してもいい？"
                        )
                    ),
                    QuickReplyButton(
                        action=MessageAction(
                            label="公式にお問い合わせする", text="お問い合わせできるところどこ？"
                        )
                    )
                ]
            )
        )
        line_bot_api.reply_message(reply_token, yet_answer_with_quick_reply)
        return

    elif incoming_text == "解決したよ！ありがとうオタ姫ちゃん！":
        yes_text = TextSendMessage(text="よかった😊、また何かあったら聞いてね🙋‍♀️💖")
        line_bot_api.reply_message(reply_token, yes_text)
        return
    
    elif incoming_text == "もう一回質問してもいい？":
        return

    loader = CSVLoader(
        file_path="app/data/questions.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["id", "question", "title"],
        },
        encoding="utf-8",
    )
    docs = loader.load()

    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    persist_directory = "chroma_faqs"
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name="faqs",
        persist_directory=persist_directory,
    )

    docs = vectordb.similarity_search_with_relevance_scores(incoming_text, k=1)
    page_content = docs[0][0].page_content
    similarity = docs[0][1]
    title = page_content.split("\ntitle: ")[1]

    if similarity < 0.7:
        default_text = TextSendMessage(text="ん〜🤔、ちょっといい答えが見つからないなあ😅聞き方を変えてくれると思い出せるかも！🔥💕")
        line_bot_api.reply_message(reply_token, default_text)
        return

    response = requests.get(url + title).json()
    descriptions = response.get("descriptions", [])

    descriptions_list: list = [
        description
        for description in descriptions
        if not question_re_pattern.search(description)
    ]
    description_text = "".join(descriptions_list)

    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {
                "role": "system",
                "content": "あなたは新入生からの質問に答える女性です。また、あなたはオタサーの姫です。メンヘラ気質で束縛が激しいです",
            },
            {
                "role": "user",
                "content": f"「{incoming_text}」という質問が来ています。データベースには似た質問として「{title}」というのがあり、その回答は「{description_text}」です.オタサーの姫風に2,3行で返してください. 文章には絵文字を多用します。この回答が質問の答えになってるか確認してください",
            },
        ],
    )
    reply_text = completion.choices[0].message.content

    text_message_with_quick_reply = TextSendMessage(
        text=reply_text,
        quick_reply=QuickReply(
            items=[
                QuickReplyButton(
                    action=MessageAction(
                        label="解決した", text="解決したよ！ありがとうオタ姫ちゃん！"
                    )
                ),
                QuickReplyButton(
                    action=MessageAction(
                        label="解決してない",
                        text="オタ姫ちゃん何言ってるかわからないよ...",
                    )
                ),
            ]
        ),
    )
    line_bot_api.reply_message(reply_token, text_message_with_quick_reply)
    return
