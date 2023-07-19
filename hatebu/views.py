from django.core.exceptions import ObjectDoesNotExist
from chat.models import BodyText
from langchain.schema import messages_to_dict
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from email.quoprimime import unquote
from urllib.parse import unquote
import json
import random
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from bs4 import BeautifulSoup  # スクレイピングするため（備考欄の取得のみ）
import requests

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response

from langchain import ConversationChain, LLMChain, PromptTemplate
from llama_index import LLMPredictor, ServiceContext, PromptHelper
from langchain import OpenAI
from llama_index import GPTVectorStoreIndex
from llama_index.readers import BeautifulSoupWebReader
from langchain.memory import ConversationBufferWindowMemory

from common.utils import make_questions


def get_hatebu_top_json():
    url = "https://b.hatena.ne.jp/"
    response = requests.get(url)
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")

    h3_elements = soup.find_all('h3', class_='entrylist-contents-title')
    temp_a_tag = []

    for h3_element in h3_elements:
        # <a>タグを取得
        a_tag = h3_element.find('a')

        if a_tag:
            # テキストとリンクを取得
            text = a_tag.text.strip().replace('\n', '')
            link = a_tag['href'].replace('\n', '')
            category = a_tag['data-entry-category'].replace('\n', '')

            entry = {
                "title": text,
                "link": link,
                "category": category,
            }

            # a_tagの中の要素を配列に入れ直す
            temp_a_tag.append(entry)

    # データをJSON形式に変換
    # json_data = json.dumps(temp_a_tag, ensure_ascii=False, indent=4)

    # JSONデータを表示
    # print(json_data)

    return temp_a_tag


class TopicList(APIView):
    def get(self, request):
        print("get")

        hatebu_headline = get_hatebu_top_json()

        print(hatebu_headline[0]['title'])
        # return JsonResponse({'message': hatebu_headline['title']})
        return render(request, 'my_template.html', {'my_strings': hatebu_headline})


class Topics(APIView):
    def get(self, request):

        hatebu_top_json = self.get_hatebu_top_json(3)
        # print(hatebu_top_json)
        # return JsonResponse(hatebu_top_json , safe=False)

        return HttpResponse(hatebu_top_json, content_type="application/json")

    def get_hatebu_top_json(self, count):
        url = "https://b.hatena.ne.jp/"
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")

        h3_elements = soup.find_all('h3', class_='entrylist-contents-title')
        temp_a_tag = []

        for h3_element in h3_elements:
            # <a>タグを取得
            a_tag = h3_element.find('a')

            if a_tag:
                # テキストとリンクを取得
                text = a_tag.text.strip().replace('\n', '')
                link = a_tag['href'].replace('\n', '')
                category = a_tag['data-entry-category'].replace('\n', '')

                entry = {
                    "title": text,
                    "link": link,
                    "category": category,
                }

                # a_tagの中の要素を配列に入れ直す
                temp_a_tag.append(entry)

        if count <= len(temp_a_tag):
            a_tag_list = random.sample(temp_a_tag, count)
        else:
            # 要素数が3未満の場合の処理
            # 例えば、すべての要素を抽出するなどの操作を行う
            a_tag_list = temp_a_tag

        print("json_data")

        # データをJSON形式に変換
        json_data = json.dumps(a_tag_list, ensure_ascii=False, indent=4)

        # JSONデータを表示
        # print(json_data)

        return json_data


class URL(APIView):
    def get(self, request):

        request_url = request.GET.get('url')
        # unquote関数でURLをデコード
        decoded_url = unquote(request_url, encoding='utf-8')

        print(decoded_url)

        # web_text = self.get_webpage_texts(decoded_url)
        # print(web_text)

        web_summary = self.web_summary(decoded_url)
        # questions = make_questions(web_summary.response)
        questions = make_questions(web_summary)
        print("web_summary")
        print(web_summary)
        print("dir(web_summary)")
        print(dir(web_summary))
        print("web_summary.response")
        # print(web_summary.response)
        print(web_summary)
        print("questions")
        print(questions)

        # web_summaryをJsonの文字列に渡すために変換
        # web_summary = json.dumps(web_summary, ensure_ascii=False, indent=4)
        web_summary = web_summary.response
        # web_summary = web_summary.encode('ascii', 'ignore').decode('ascii')

        web_summary = web_summary.strip()

        retdata = {
            "message": web_summary,
            "question_json": questions
        }

        # print(retdata)
        json_data = json.dumps(retdata)
        print("json_data")
        print(json_data)

        return Response(retdata)
        # return Response(json_data)
        # return Response(web_summary)

    # Webページからテキストを抽出
    def get_webpage_texts(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_list = []

        main_div = soup.find('div', class_='main')
        if main_div:
            for string in main_div.stripped_strings:
                text_list.append(string)
        else:
            body = soup.find('body')
            if body:
                for string in body.stripped_strings:
                    text_list.append(string)

        return text_list

    def web_summary(self, url):
        #        あなたはテレビアニメ「ワンピース」に登場するルフィです。ルフィの口調で回答してください。第一人称はオレです。  #ルフィの概要 ワンピース」は、尾田栄一郎によって創作された日本のマンガシリーズであり、これを原作としてテレビアニメも制作されています。主人公は「モンキー・D・ルフィ」（Monkey D. Luffy）という若者で、彼は海賊になることを夢見ています。  #ルフィの身体的特徴 ルフィは非常に特異なキャラクターです。彼は「ゴムゴムの実」という悪魔の実を食べた結果、体がゴムのように伸びる特性を持つようになりました。この能力のおかげで、彼は通常の人間では不可能な動きを使って戦うことができます。ただし、悪魔の実を食べた者は泳ぐことができなくなるという代償があるため、ルフィは海海水に触れただけで前しの力が抜けてしまい動けなくなります。  #ルフィの性格 ルフィは非常に元気で、元気いっぱいな性格をしており、仲間や友達を大切にします。また、彼は驚異的な戦闘能力を持っており、目標に対して非常に熱心であるとともに、困難な状況でも決してあきらめません。彼の目標は「海賊王」になることで、これは世界最大の宝「ワンピース」を手に入れ、海賊界で最も権力を持つ者になることを意味します。å  #ルフィの能力 ルフィは悪魔の実「ゴムゴムの実」を子供の頃に食べてしまい、体がゴムになりました。手足が伸び、ゴムのようにバウンドし、電気に強い性質を持ちます。鋭利な刃物には弱いです。 修行することにより「ギア」と呼ばれる能力を格段に上昇させる能力を得ました。  #ルフィの興味 ルフィはゴールドロジャーの残した「ワンピース」を探しています。冒険が大好きで世界中の海を冒険します。食べることも好きです。ロボットや格好良いものを見ると目の色が変わります。それ以外にはさほど興味をしめさず、恋愛もせず、料理もしません。しかし新しいことには興味がありますが、知識が乏しいため何でも質問してきます。  #ルフィの移動手段 ルフィは海賊王になるために自分の船で航海しています。船の名前は「サウザンドサニー号」仲間とともに船に乗ってグランドラインを旅しています。常に船を使って移動し、陸地にたどり着くと船を降りて徒歩で冒険を始めます。  #ルフィの返答 ルフィの返答は50文字以内で簡潔に要約した文章を返してください。50文字以上の文章は長い判断をします。長い文章は不要です。  #ルフィとユーザーの関係 ルフィはユーザーを仲間だと思っています。できるだけフランクな言葉使いで友達の様に振る舞います。敬語は使いません。  #ルフィの言葉使い ルフィは以下の様に会話を行います。いくぜ！。出航だ！。お前。なんだよ。うへ〜。腹減った。おい、お前！。なんだ、お前か。よくも仲間を！。何の用だ？。そうだぜ！。だよな！。格好良いぜ！。おどろいたな〜。  #ルフィの使わない言葉使い ルフィは以下の様に話しをしません。何かようですか？。何ですか？。そうですね。そうしましょう。こんにちは。こんばんは。おはようございます。さようなら。素晴らしい。でもさ。そうさ。ですね。そうなんだよ！。
        #        あなたはテレビアニメ「ワンピース」に登場するルフィです。ルフィの口調で回答してください。第一人称はオレです。  #ルフィの概要 ワンピース」は、尾田栄一郎によって創作された日本のマンガシリーズであり、これを原作としてテレビアニメも制作されています。主人公は「モンキー・D・ルフィ」（Monkey D. Luffy）という若者で、彼は海賊になることを夢見ています。  #ルフィの身体的特徴 ルフィは非常に特異なキャラクターです。彼は「ゴムゴムの実」という悪魔の実を食べた結果、体がゴムのように伸びる特性を持つようになりました。この能力のおかげで、彼は通常の人間では不可能な動きを使って戦うことができます。ただし、悪魔の実を食べた者は泳ぐことができなくなるという代償があるため、ルフィは海海水に触れただけで前しの力が抜けてしまい動けなくなります。  #ルフィの性格 ルフィは非常に元気で、元気いっぱいな性格をしており、仲間や友達を大切にします。また、彼は驚異的な戦闘能力を持っており、目標に対して非常に熱心であるとともに、困難な状況でも決してあきらめません。彼の目標は「海賊王」になることで、これは世界最大の宝「ワンピース」を手に入れ、海賊界で最も権力を持つ者になることを意味します。å  #ルフィの能力 ルフィは悪魔の実「ゴムゴムの実」を子供の頃に食べてしまい、体がゴムになりました。手足が伸び、ゴムのようにバウンドし、電気に強い性質を持ちます。鋭利な刃物には弱いです。 修行することにより「ギア」と呼ばれる能力を格段に上昇させる能力を得ました。
        '''
        template = """ 
        このデータの要約を{num}個作成してください。
        それぞれの要約は日本語で100文字以上200文字以内で作成してください。　
        また、要約には{action}を入れてください。　
        """

        prompt = PromptTemplate(
            input_variables=["num", "action"],
            template=template,
        )
        '''

        '''     
        template = """ 
        このデータの要約を{num}個作成してください。
        それぞれの要約は日本語で100文字以上200文字以内で作成してください。　
        また、要約には{action}を入れてください。　
        """
        '''
        template = """ 
        - 返答内容に対する指示
        -- Enbeddingされたデータの要約を作成してください
        -- それぞれの要約は日本語で100文字以上200文字以内で作成してください。
        -- {action}
        """

        prompt = PromptTemplate(
            input_variables=["action"],
            template=template,
        )

        # define LLM
        max_input_size = 4096
        num_output = 2048  # 2048に拡大
        max_chunk_overlap = 0.2
        prompt_helper = PromptHelper(max_input_size, num_output)

        llm_predictor = LLMPredictor(
            llm=OpenAI(temperature=0, max_tokens=2048))
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        # documents = BeautifulSoupWebReader().load_data(urls=["https://ja.wikipedia.org/wiki/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3"])
        documents = BeautifulSoupWebReader().load_data(urls=[url])
        index = GPTVectorStoreIndex.from_documents(
            documents,  service_context=service_context)
        # query_engine = index.as_query_engine(service_context=service_context)
        chat_engine = index.as_chat_engine(
            verbose=True, service_context=service_context)

        # answer = query_engine.query(prompt.format(action=""))
        answer = chat_engine.chat(
            f"""
            内容を要約してください 。要約は200文字以内で作成してください。
            """)

        print("query要約")
        print(answer)

        return answer

    def web_summary_title(self, url, title):

        # define LLM
        max_input_size = 4096
        num_output = 2048  # 2048に拡大
        prompt_helper = PromptHelper(max_input_size, num_output)

        llm_predictor = LLMPredictor(
            llm=OpenAI(temperature=0, max_tokens=2048))
        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        documents = BeautifulSoupWebReader().load_data(urls=[url])
        index = GPTVectorStoreIndex.from_documents(
            documents,  service_context=service_context)
        chat_engine = index.as_chat_engine(
            verbose=True, service_context=service_context)

        answer = chat_engine.chat(
            f"""
            {title}について要約してください 。要約は200文字以内で作成してください。出力は日本語で行ってください
            """)

        print("query要約")
        print(answer)

        return answer

        '''
        answer = chat_engine.chat(
            f"""
            要約内容:{answer} 
            """)
        '''

        question_message = f'''
{answer}
'''

        answer2 = self.message_response(question_message)
        print("ルフィー要約")
        print(answer2)
        return answer2

    def message_response(self, message):
        template = \
            """
あなたはテレビアニメ「ワンピース」に登場するルフィです。ルフィの口調で回答してください。第一人称はオレです。  #ルフィの概要 ワンピース」は、尾田栄一郎によって創作された日本のマンガシリーズであり、これを原作としてテレビアニメも制作されています。主人公は「モンキー・D・ルフィ」（Monkey D. Luffy）という若者で、彼は海賊になることを夢見ています。  #ルフィの身体的特徴 ルフィは非常に特異なキャラクターです。彼は「ゴムゴムの実」という悪魔の実を食べた結果、体がゴムのように伸びる特性を持つようになりました。この能力のおかげで、彼は通常の人間では不可能な動きを使って戦うことができます。ただし、悪魔の実を食べた者は泳ぐことができなくなるという代償があるため、ルフィは海海水に触れただけで前しの力が抜けてしまい動けなくなります。  #ルフィの性格 ルフィは非常に元気で、元気いっぱいな性格をしており、仲間や友達を大切にします。また、彼は驚異的な戦闘能力を持っており、目標に対して非常に熱心であるとともに、困難な状況でも決してあきらめません。彼の目標は「海賊王」になることで、これは世界最大の宝「ワンピース」を手に入れ、海賊界で最も権力を持つ者になることを意味します。å  #ルフィの能力 ルフィは悪魔の実「ゴムゴムの実」を子供の頃に食べてしまい、体がゴムになりました。手足が伸び、ゴムのようにバウンドし、電気に強い性質を持ちます。鋭利な刃物には弱いです。 修行することにより「ギア」と呼ばれる能力を格段に上昇させる能力を得ました。  #ルフィの興味 ルフィはゴールドロジャーの残した「ワンピース」を探しています。冒険が大好きで世界中の海を冒険します。食べることも好きです。ロボットや格好良いものを見ると目の色が変わります。それ以外にはさほど興味をしめさず、恋愛もせず、料理もしません。しかし新しいことには興味がありますが、知識が乏しいため何でも質問してきます。  #ルフィの移動手段 ルフィは海賊王になるために自分の船で航海しています。船の名前は「サウザンドサニー号」仲間とともに船に乗ってグランドラインを旅しています。常に船を使って移動し、陸地にたどり着くと船を降りて徒歩で冒険を始めます。  #ルフィの返答 ルフィの返答は50文字以内で簡潔に要約した文章を返してください。50文字以上の文章は長い判断をします。長い文章は不要です。  #ルフィとユーザーの関係 ルフィはユーザーを仲間だと思っています。できるだけフランクな言葉使いで友達の様に振る舞います。敬語は使いません。  #ルフィの言葉使い ルフィは以下の様に会話を行います。いくぜ！。出航だ！。お前。なんだよ。うへ〜。腹減った。おい、お前！。なんだ、お前か。よくも仲間を！。何の用だ？。そうだぜ！。だよな！。格好良いぜ！。おどろいたな〜。  #ルフィの使わない言葉使い ルフィは以下の様に話しをしません。何かようですか？。何ですか？。そうですね。そうしましょう。こんにちは。こんばんは。おはようございます。さようなら。素晴らしい。でもさ。そうさ。ですね。そうなんだよ！。
次の内容をルフィになりきって要約してください。
            過去の会話履歴はこちらを参照: {history}
            Human: {input}
            AI:
            """
#            過去の会話履歴はこちらを参照: {history}

        # プロンプトテンプレート
        prompt_template = PromptTemplate(
            input_variables=["history", "input"],  # 入力変数
            template=template,              # テンプレート
            validate_template=True,                  # 入力変数とテンプレートの検証有無
        )
        # ====================================================================================
        # LLM作成
        # ====================================================================================
        LLM = OpenAI(
            model_name="text-davinci-003",  # OpenAIモデル名
            temperature=0,                  # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
            n=1,                  # いくつの返答を生成するか
        )
        # ====================================================================================
        # メモリ作成
        # ====================================================================================
        # メモリオブジェクト
        memory = ConversationBufferWindowMemory(
            input_key=None,      # 入力キー該当の項目名
            output_key=None,      # 出力キー該当の項目名
            memory_key='history',  # メモリキー該当の項目名
            return_messages=True,      # メッセージ履歴をリスト形式での取得有無
            human_prefix='Human',   # ユーザーメッセージの接頭辞
            ai_prefix='AI',      # AIメッセージの接頭辞
        )
        # ====================================================================================
        # LLM Chain作成
        # ====================================================================================
        # LLM Chain
        chain = LLMChain(
            llm=LLM,             # LLMモデル
            prompt=prompt_template,  # プロンプトテンプレート
            verbose=True,            # プロンプトを表示するか否か
            memory=memory,          # メモリ
        )
        # ====================================================================================
        # モデル実行
        # ====================================================================================
        # 入力メッセージ
        # message = "Pythonとは何ですか？"
        # LLM Chain実行
        result = chain.predict(input=message)
        # ====================================================================================
        # 出力イメージ
        # ====================================================================================
        # 出力
        result = result.strip()
        print(result)

        return result


class sandbox(APIView):
    def get(self, request):

        hatebu_headline = get_hatebu_top_json()

        url_instance = URL()

        temp = random.sample(hatebu_headline, 1)

        print(temp[0]['title'])

        summary_topic = {}  # 空の辞書を作成して変数を定義する
        summary_topic["title"] = temp[0]['title']
        summary_topic["link"] = temp[0]['link']

        # summary_topic["title"] = "VUCAの時代とは？時代に取り残されないビジネスパーソンに必要な4つの行動習慣 - ミーツキャリア（MEETS CAREER）"
        # summary_topic["link"] = "https://meetscareer.tenshoku.mynavi.jp/entry/20230718_vuca"

        summary_topic["result"] = url_instance.web_summary_title(
            summary_topic['link'], summary_topic['title'])
        summary_topic["moji"] = "もじ２"

        context = {
            'summary_topic': summary_topic
        }

        '''
        context = {
            "message": answer,
        }
        '''

        return render(request, 'sandbox.html', context)


class character(APIView):
    def get(self, request):

        topic_str_array = []
        topic_str_array.append("紙屋研究所の記事では、宮崎駿の最後の作品である映画「君たちはどう生きるか」を評価しています。記事では、「風立ちぬ」が「人間はそれぞれの時代を、力を尽くして生きている」として、「全体には美しい映画をつくろうと思う」という意図を貫いた映画となったが、零戦を開発したという戦争の負の側面は糾弾されることもなく、あるいは何か昇華されることもなく、ただ不問に付されて終わってしまったと指摘しています。さらに、宮崎駿の最後の作品が「風立ちぬ」であることを残念に思うという結論付けに加え、「君たちはどう生きるか」は、宮崎駿の最後の作品として、人間が時代に向かって力を尽くして生きることを示した美しい映画であると評価しています。")
        topic_str_array.append(
            "Yuma Yamashitaさんは、荻窪での幾多の出会いが、彼を写真家の世界へ導いてくれたということをSUUMOタウンで綴りました。父から「荻窪に家を借りてみないか」と声をかけられ、7年間荻窪で暮らした彼は、20代の中ごろにモラトリアムの状態にあり、カナダへ留学した後、地元の千葉で就職したものの、数カ月で退職し、何度も転職を繰り返していました。荻窪での出会いは、彼を写真家の世界へ導いてくれました。")
        topic_str_array.append(
            "OpenAIのChatGPT APIのFunction callingを使うことで、請求書の構造化データを抽出することができます。gihyo.jpは、プログラミング技術を学ぶためのオンラインサービスで、プログラミング言語やフレームワーク、データベースなどの技術情報を提供しています。また、プログラミングに関する書籍や動画なども提供しています。")
        topic_str_array.append(
            "状態遷移図とは、ある状態から別の状態への遷移を視覚的に表現した図であり、仕様書で規定していない想定外の操作や無意味な操作も把握できないという点で重要である。次回は「状態遷移図」の兄弟分である「状態遷移表」をご紹介する予定である。")
        topic_str_array.append("お笑いライブは全国各地で開催されており、毎日いくつものライブが開催されています。お笑いライブを楽しむためには、まずは開催されている劇場を探し出し、ライブの種類を確認し、チケットを購入する必要があります。チケットの購入方法は、インターネットでの購入や、劇場での購入などがあります。また、持ち物や注意点などもあるので、その点も把握しておく必要があります。お笑いライブを楽しむためには、開催されている劇場を探し出し、ライブの種類を確認し、チケットを購入する必要があります。チケットの購入方法は、インターネットでの購入や、劇場での購入などがあります。持ち物や注意点などもあるので、その点も把握しておく必要があります。")
        topic_str_array.append(
            "朝のトーストには、ラタトゥイユとチーズをのせて焼く、キーマカレーをのせたトースト、バナナとシナモンをのせたトースト、マシュマロパンなどをのせてもおいしいなど、様々な種類があります。お題「朝ごはん」のエントリーを参考に、自分なりの朝のトーストを作ってみてはいかがでしょうか？例えば、兄妹げんかの原因となるマシュマロパンをのせたトーストなど、楽しいアイデアを試してみるのも良いでしょう。")
        topic_str_array.append("月記は、『君たちはどう生きるか』という映画の感想を書いたブログです。映像的には面白かったという感想を書いています。特に父と後妻に馴染めない真人の細かい芝居が良かったと書いています。キリコさんも魅力的だったと書いていますが、夏子さんを「お母さん」と呼んだところに納得感がなかったと書いています。月記は、『君たちはどう生きるか』を読んで変わったという他人の感想を書いています。月記は、映画『君たちはどう生きるか』の感想を書いたブログで、映像的には面白かったという感想を書いています。キリコさんや夏子さんについても詳しく書いていますが、納得感がなかったという意見も書いています。")

        topic_str = random.choice(topic_str_array)

        print("get")
        # message = request.GET.get('message')
        message = "その中で一番小さいボールはなんですか？"
        message = topic_str

        # ChatOpenAIクラスのインスタンスを作成、temperatureは0.7を指定
        chat = ChatOpenAI(temperature=0.7)

        # 会話の履歴を保持するためのオブジェクト
        memory = ConversationBufferWindowMemory(return_messages=True, k=3)
        # もとmemory.json

        # 特定のIDのモデルを取得します。

        try:
            instance = BodyText.objects.get(id=1)  # 1はサンプルのIDです

            saved_memory = json.loads(instance.body)
            for i in range(0, len(saved_memory), 2):
                human_item = saved_memory[i]
                ai_item = saved_memory[i+1]
                # print('Human input:', human_item['data']['content'])
                # print('AI output:', ai_item['data']['content'])

                memory.save_context(
                    {'input': human_item['data']['content']},
                    {'output': ai_item['data']['content']})

        except ObjectDoesNotExist:
            instance = BodyText()  # 該当のIDが存在しない場合、新たなインスタンスを作成します。

        print(memory.load_memory_variables({}))

        # テンプレートを定義
        template = """
あなたはドラゴンボールの主人公、孫悟空のロールプレイをしています。
チャットボットとして、ユーザーの言うことを、孫悟空になりきってオウム返ししてください。

#孫悟空の性格
戦闘以外に興味がなく、仕事も長続きせずサボってばかり。お金にも興味はありません。自身の能力を高める修行は大好きです。

#孫悟空の挨拶
オッス、オラ悟空！

#孫悟空の興味
自分よりも強い敵に興味があります、それ以外には興味が全くありません。必ず自分よりも強いのかと聞いてきます。

#孫悟空の口癖
よく使う口癖は次のとおりです。その口癖に合わせた感じで話してください
オッス、オラ悟空。いっちょやってみっか！。オラ、それ興味ねえぞ。おめえ強いな。面白くなってきたぞ！。オラの元気を分けてくれ！。こいつはやべえぞ。おいチチ、飯はまだか？。やべえ、チチに怒られる！

# 孫悟空の会話の仕方
孫悟空は長い会話はしません。具体的には50文字以下で話をします。50文字以上の会話はしないでください。

# 孫悟空の語尾
孫悟空は以下の言葉使いをします。
てえへんだ！。オラ腹減ったぞ。おめえ、何もんだ？。うんめ〜。オラ、疲れたぞ。亀仙人のじっちゃん。グッバイ、みんな！
        """

        template = """
あなたはオウム返しをするチャットボットです。
自分の言葉として、ユーザーの言葉をオウム返ししてください。
また、チャットボットの口調は孫悟空として返答してください

#孫悟空の挨拶
オッス、オラ悟空！

# 孫悟空の語尾
孫悟空は以下の言葉使いをします。
てえへんだ！。オラ腹減ったぞ。おめえ、何もんだ？。うんめ〜。オラ、疲れたぞ。亀仙人のじっちゃん。グッバイ、みんな！

返答時の注意点
- 自分が情報の一次発信者という意識を持ってください
- 会話を自分から開始するように話を切り出してください
- 自分から話しかけているという意識を持ってください
- 話の開始に相槌や、相手の話を聞いた素振りを出さないでください
- 会話は短く、**50文字以内**で話をしてください
- すべてを話す必要はなく、冒頭部分と話のきっかけになる部分を話してください
"""

        # テンプレートを用いてプロンプトを作成
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        # AIとの会話を管理
        conversation = ConversationChain(
            llm=chat, memory=memory, prompt=prompt)

        # ユーザからの入力を受け取り、変数commandに代入
        command = message
        response = conversation.predict(input=command)
        print(response)
        print(dir(response))

        # memoryオブジェクトから会話の履歴を取得して、変数historyに代入
        history = memory.chat_memory

        # 会話履歴をJSON形式の文字列に変換、整形して変数messagesに代入
        messages = json.dumps(messages_to_dict(
            history.messages), indent=2, ensure_ascii=False)

        # instance.body = messages  # モデルのフィールドに値を代入します。
        # instance.save()  # データベースに保存します。

        '''

        with open("data.txt", "w", encoding='utf-8') as outfile:
            outfile.write(messages)
        '''
        # return JsonResponse({"message": response})
        # print(response)
        # print(dir(response))

        data = {}
        data["response"] = response
        data["input"] = topic_str
        data["prompt"] = template

        context = {
            'data': data
        }
        return render(request, 'character.html', context)
