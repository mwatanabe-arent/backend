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

from langchain import LLMChain, PromptTemplate
from llama_index import LLMPredictor, ServiceContext, PromptHelper
from langchain import OpenAI
from llama_index import GPTVectorStoreIndex
from llama_index.readers import BeautifulSoupWebReader
from langchain.memory import ConversationBufferWindowMemory

from common.utils import make_questions


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
        # web_summary = web_summary.response
        # web_summary = web_summary.encode('ascii', 'ignore').decode('ascii')

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

        template = """ 
        このデータの要約を{num}個作成してください。
        それぞれの要約は日本語で100文字以上200文字以内で作成してください。　
        また、要約には{action}を入れてください。　
        """

        prompt = PromptTemplate(
            input_variables=["num", "action"],
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
        query_engine = index.as_query_engine(service_context=service_context)
        chat_engine = index.as_chat_engine(service_context=service_context)

        answer = query_engine.query(prompt.format(num=3, action="Webページの要約"))
        print("query要約")
        print(answer)
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
