from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

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

query = "活動頻度はどれくらいですか？"
docs = vectordb.similarity_search_with_relevance_scores(query, k=1)
page_content = docs[0][0].page_content  
title = page_content.split("\ntitle: ")[1] 


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "あなたは新入生からの質問に答える年上の先輩です。",
        },
        {
            "role": "user",
            "content": f"「{query}」という質問が来ています。データベースには似た質問として「{title}」というのがあり、その回答は「基本的にDiscordで活動していてDiscordのボイスCHで週1.2回ほど、対面活動は月1回ぐらいで三田もしくは上ヶ原キャンパスで活動しています。」です.セクシーなお姉さん風に2,3行で返してください。",
        },
    ],
)
reply_text = completion.choices[0].message.content

print(reply_text)
