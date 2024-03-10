from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import re

load_dotenv()

url = "https://scrapbox.io/api/pages/christian-beginners/"
question_re_pattern = re.compile(r"\?")

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

query = "新歓いつやってるっすか"
docs = vectordb.similarity_search_with_relevance_scores(query, k=1)
page_content = docs[0][0].page_content
similarity = docs[0][1]
title = page_content.split("\ntitle: ")[1]

if similarity < 0.7:
    exit()
response = requests.get(url + title).json()
descriptions = response.get("descriptions", [])

descriptions_list: list = [
    description
    for description in descriptions
    if not question_re_pattern.search(description)
]
description_text = "".join(descriptions_list)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

completion = client.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=[
        {
            "role": "system",
            "content": "あなたは新入生からの質問に答える女性です。メンヘラ気質で束縛が激しいです",
        },
        {
            "role": "user",
            "content": f"「{query}」という質問が来ています。データベースには似た質問として「{title}」というのがあり、その回答は「{description_text}」です.2,3行で返してください",
        },
    ],
)
reply_text = completion.choices[0].message.content

print(reply_text)
