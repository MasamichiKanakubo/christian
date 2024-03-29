# import csv
# import json

# csv_file_path = "app/data/questions.csv"

# json_data = [
#     {"パソコンの貸し出しはありますか？": "パソコンの貸し出しはありますか？"},
#     {"パソコンの貸与はありますか？": "パソコンの貸し出しはありますか？"},
#     {"PCの貸し出しはありますか？": "パソコンの貸し出しはありますか？"},
#     {"PCの貸与はありますか？": "パソコンの貸し出しはありますか？"},
#     {"新歓はありますか？": "新歓はあるか"},
#     {"新歓はいつですか？": "新歓はあるか"},
#     {"歓迎会はありますか？": "新歓はあるか"},
#     {"歓迎会はいつですか？": "新歓はあるか"},
#     {"新入生歓迎会はありますか？": "新歓はあるか"},
#     {"社会人でも参加できますか？": "社会人でも参加できますか？"},
#     {"社会人でも入れますか？": "社会人でも参加できますか？"},
#     {"学生以外でも参加できますか？": "社会人でも参加できますか？"},
#     {"学生以外でも入れますか？": "社会人でも参加できますか？"},
#     {"友達はできますか？": "友達はできますか？"},
#     {"友達ができますか？": "友達はできますか？"},
#     {"友人はできますか？": "友達はできますか？"},
#     {"友人ができますか？": "友達はできますか？"},
#     {"大学生活を楽しめますか？": "友達はできますか？"},
#     {"大学生活を謳歌できますか": "友達はできますか？"},
#     {"学生生活を楽しめますか？": "友達はできますか？"},
#     {"学生生活を謳歌できますか？": "友達はできますか？"},
#     {"サークルへの参加は何か特定のスキルや経験が必要ですか？": "サークル参加の条件"},
#     {"サークルへの参加は何か特定のスキルや経験がいりますか？": "サークル参加の条件"},
#     {"サークルへの参加は何か特定の条件が必要ですか？": "サークル参加の条件"},
#     {"サークルへの参加は何か特定の条件がいりますか？": "サークル参加の条件"},
#     {"サークルの指導者はいますか？": "サークルの指導者は誰ですか？"},
#     {"サークルの指導者は誰？": "サークルの指導者はいますか？"},
#     {"サークルの顧問はいますか？": "サークルの指導者はいますか？"},
#     {"サークルの顧問は誰ですか？": "サークルの指導者はいますか？"},
#     {"サークルの顧問は誰？": "サークルの指導者はいますか？"},
#     {"初心者でも大丈夫か？": "初心者でも大丈夫か"},
#     {"初心者でもついていけるか？": "初心者でも大丈夫か"},
#     {"初心者でも問題ないか？": "初心者でも大丈夫か"},
#     {"はじめたてでも大丈夫か？": "初心者でも大丈夫か"},
#     {"はじめたてでもついていけるか？": "初心者でも大丈夫か"},
#     {"はじめたてでも問題ないか？": "初心者でも大丈夫か"},
#     {"Tech.Uniの歴史について": "Tech.Uniとは何ですか？"},
#     {"Tech.Uniの歴史とは": "Tech.Uniとは何ですか？"},
#     {"techuniの歴史について": "Tech.Uniとは何ですか？"},
#     {"techuniの歴史とは": "Tech.Uniとは何ですか？"},
#     {"techuniの活動内容について": "Tech.Uniとは何ですか？"},
#     {"techuniの活動内容とは": "Tech.Uniとは何ですか？"},
#     {"テックユニの歴史について": "Tech.Uniとは何ですか？"},
#     {"テックユニの歴史とは": "Tech.Uniとは何ですか？"},
#     {"テックユニの活動内容について": "Tech.Uniとは何ですか？"},
#     {"テックユニの活動内容とは": "Tech.Uniとは何ですか？"},
#     {"techuniの活動内容について": "Tech.Uniとは何ですか？"},
#     {"techuniの活動内容とは": "Tech.Uniとは何ですか？"},
#     {"テックユニの歴史について": "Tech.Uniとは何ですか？"},
#     {"テックユニの活動内容について": "Tech.Uniとは何ですか？"},
#     {"テックユニの活動内容とは": "Tech.Uniとは何ですか？"},
#     {"テクユニの歴史について": "Tech.Uniとは何ですか？"},
#     {"テクユニの歴史とは": "Tech.Uniとは何ですか？"},
#     {"テクユニの活動内容について": "Tech.Uniとは何ですか？"},
#     {"テクユニの活動内容とは": "Tech.Uniとは何ですか？"},
#     {"情報学部じゃなくても入れますか？": "情報学部じゃなくても入れますか"},
#     {"情報専攻じゃなくても入れますか？": "情報学部じゃなくても入れますか"},
#     {"ハッカソンとはなんですか?": "ハッカソンとは"},
#     {"ハッカソンとはどんなイベント?": "ハッカソンとは"},
#     {
#         "プログラミングしたことがないのですが大丈夫ですか？": "プログラミング経験なしでも入れるか"
#     },
#     {
#         "プログラミングしたことがないのですが問題ないですか？": "プログラミング経験なしでも入れるか"
#     },
#     {
#         "プログラミング経験がないのですが大丈夫ですか？": "プログラミング経験なしでも入れるか"
#     },
#     {
#         "プログラミング経験がないのですが問題ないですか？": "プログラミング経験なしでも入れるか"
#     },
#     {"参加人数は何人ですか?": "参加人数は何人"},
#     {"参加人数は何人いますか?": "参加人数は何人"},
#     {"在籍人数は何人ですか?": "参加人数は何人"},
#     {"在籍人数は何人いますか?": "参加人数は何人"},
#     {"他大学でも参加できますか？": "他大学でも参加できますか"},
#     {"他大学でも加入できますか？": "他大学でも参加できますか"},
#     {"このサークルの強みはなんですか?": "強み・メリットはなに？"},
#     {"このサークルのメリットはなんですか?": "強み・メリットはなに？"},
#     {"このサークルの良さはなんですか?'": "強み・メリットはなに？"},
#     {"会費はいくらですか？": "会費はいくら"},
#     {"会費はかかりますか？": "会費はいくら"},
#     {"入会費はいくらですか？": "会費はいくら"},
#     {"入会費はかかりますか？": "会費はいくら"},
#     {"参加費はいくらですか？": "会費はいくら"},
#     {"参加費はかかりますか？": "会費はいくら"},
#     {"お金はいくらですか？": "会費はいくら"},
#     {"お金はかかりますか？": "会費はいくら"},
#     {"男女比はどの程度ですか？": "男女比はどの程度か"},
#     {"男女比はどのくらいですか？": "男女比はどの程度か"},
#     {"男女比率はどの程度ですか？": "男女比はどの程度か"},
#     {"男女比率はどのくらいですか？": "男女比はどの程度か"},
#     {"どれぐらいの頻度で活動していますか?": "どれぐらいの頻度で活動していますか"},
#     {"どれぐらいのペースで活動していますか?": "どれぐらいの頻度で活動していますか"},
#     {"どのぐらいの頻度で活動していますか?": "どれくらいの頻度で活動していますか"},
#     {"どのぐらいのペースで活動していますか?": "どれくらいの頻度で活動していますか"},
# ]

# # CSVファイルへの書き込み
# with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.DictWriter(file, fieldnames=["id", "question", "title"])
#     writer.writeheader()

#     for i, item in enumerate(json_data, start=1):
#         for question, title in item.items():
#             writer.writerow({"id": i, "question": question, "title": title})
