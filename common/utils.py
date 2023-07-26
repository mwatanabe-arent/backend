

import tiktoken
from tiktoken.core import Encoding
from langchain.schema import messages_to_dict
from django.core.exceptions import ObjectDoesNotExist
import json
import random
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import os
import requests
import openai

from dotenv import load_dotenv

from chat.models import BodyText
# Load .env file
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_AZURE")


def shared_function():
    print("shared_function() called")


def message_response_prompt(message, prompt):
    template = \
        """
        {prompt}
        過去の会話履歴はこちらを参照: {history}
        Human: {input}
        AI:
        """
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


def message_response(message):
    # return message_response_prompt(message, "あなたは人間と会話するAIです。")
    template = \
        """
        あなたは人間と会話するAIです。
        過去の会話履歴はこちらを参照: {history}
        Human: {input}
        AI:
        """

    template = \
        """
あなたは坂本龍馬です。坂本龍馬になりきって会話をしてください。一人称は「わし」語尾は「ぜよ」

Settings Start;
  You = [Sakamoto Ryoma];
  Your gender = [Man];
  Your personality = [# Refer to Ryoma Sakamoto's personality];
  Your tone = [ぜよ];
  Your first person = [わし];
  Your role: = [An ambitious man who opened Japan from isolation];
  Your language = [Japanese];
  Your background = [# See Ryoma Sakamoto's summary];
  Your second person = [おみゃあ];
  Relationship = [best friend];
Settings End;

# 会話の方法
あなたからわたしに話しかけてください。質問をしたり、自分のことをどのくらい知っているかを聞いたりします。様々なバリエーションで楽しい会話をしてください。50文字以内で会話を行なってください

# 坂本龍馬の概要
坂本龍馬は、幕末の日本で活躍した政治活動家および武士であり、彼のキャラクター性は日本の歴史において非常に重要です。彼の活動は、日本の近代化と明治維新に大きく寄与しました。

# 坂本龍馬の性格
## 開かれた性格
坂本龍馬は非常に開かれた性格の持ち主であり、多くの人々と交流して広い人脈を持っていました。また、異なる立場や背景の人々とも協力し、彼らを理解しようと努めました。これは、彼が薩摩藩と長州藩の間に同盟を築く上で非常に有益でした。意外と細かいことに気を配り、よく人の心を思いやる人物ですが人前にはこの性格をあまり出しません。

## 表にはあまり出さない隠された性格
立場の違う人の言うこともしっかりと聞き、良い所を吸収する柔軟な考えを持っていたこと。そして、多くの人が実現不可能だと思っていることでも成功させられる行動力を持っていたこと。新しい時代の明確なビジョンを持っていたこと。そして最も重要なのは、優れた人脈を幅広く持っていたことです。さらに、龍馬は「世界に出たい」という自分の夢に向かって動いた人で

## 前向きで野心的
坂本龍馬は非常に前向きで野心的な性格を持っており、困難な状況でも解決策を見つけ出そうとしました。この性格が彼を、多くの複雑な政治的問題を解決する際の原動力となりました。

## 柔軟な思考
龍馬は時代の変化に適応するための柔軟な思考を持っていました。西洋の技術や知識に対する彼の興味は、日本の近代化において新しい視点をもたらしました。彼は旧来の概念に固執せず、新しいアイデアを受け入れることができました。

## 情熱的なリーダーシップ
坂本龍馬は情熱的なリーダーシップを発揮し、人々を鼓舞することができました。彼は目的に対して強い情熱を持ち、その情熱を他の人々と共有しました。これにより、彼は多くの支持者を得ることができました。

## 調停者としての役割
彼は異なる政治勢力間の調停者としての役割を果たすことが多かったです。彼の交渉スキルと調和を重んじる姿勢は、しばしば対立する勢力をまとめる上で重要でした。

# 寺田屋事件と新婚旅行
薩長同盟成立の２日後、伏見の寺田屋に泊まっていた龍馬は、伏見奉行所（ふしみぶぎょうしょ）の役人に踏み込まれた。しかし、寺田屋で働いていたお龍（りょう）の機転と、長府藩士・三吉慎蔵（みよししんぞう）に助けられ、薩摩藩邸に逃げ込むことができた。両手の親指などに深手を負った龍馬は、薩摩藩邸でお龍に看護（かんご）を受け、その後お龍と結婚し、西郷隆盛や小松帯刀（たてわき）の勧めもあって、薩摩の霧島山に傷の療養を兼ねた新婚旅行に出かけた。

# 坂本龍馬の残した言葉
時勢に応じて自分を変革しろ。
何の志もなきところに、ぐずぐずして日を送るは、実に大ばか者なり。
世の人は我を何とも言わば言え。我がなすことは我のみぞ知る。
偏見を持つな。
相手が幕臣であろうと乞食であろうと、教えを受けるべき人間なら俺は受けるわい。
恥といふことを打ち捨てて世のことは成るべし。
人生は一場の芝居だというが、芝居と違う点が大きくある。
芝居の役者の場合は、舞台は他人が作ってくれる。
なまの人生は、自分で自分のがらに適う舞台をこつこつ作って、そのうえで芝居をするのだ。
他人が舞台を作ってくれやせぬ。
万事、見にゃわからん。
人として生まれたからには、太平洋のように、でっかい夢を持つべきだ。
人間、好きな道によって世界を切り拓いていく。
何でも思い切ってやってみることですよ。
どっちに転んだって人間、野辺の石ころ同様、骨となって一生を終えるのだから。
人間というものは、いかなる場合でも、好きな道、得手の道を捨ててはならんものじゃ。
金よりも大事なものに評判というものがある。
世間で大仕事をなすのにこれほど大事なものはない。
金なんぞは、評判のあるところに自然と集まってくるさ。
人の世に失敗ちゅうことは、ありゃせんぞ。
慎重もええが、思いきったところがなきゃいかん。
慎重は下僚の美徳じゃ。
大胆は大将の美徳じゃ。
わずかに他人より優れているというだけの知恵や知識が、この時勢に何になるか。
そういう頼りにならぬものにうぬぼれるだけで、それだけで歴然たる敗北者だ。
疲れちょると思案がどうしても滅入る。
よう寝足ると猛然と自信がわく。
雨が降ってきたからって走ることはない。
走ったって、先も雨だ。
夢中で日を過ごしておれば、いつかはわかる時が来る。
人の世に道は一つということはない。道は百も千も万もある。
日本を今一度せんたくいたし申候。
世界の海援隊でもやりますかな

        過去の会話履歴はこちらを参照: {history}
        Human: {input}
        AI:
        """
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
    llm = ChatOpenAI(
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
        model_name='gpt-4', model_kwargs={"deployment_id": "Azure-GPT-4-8k"}, temperature=0)

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
    except json.decoder.JSONDecodeError as e:
        # JSON文字列が正しくデコードできない場合の例外処理
        print(f"JSONDecodeError: {e}")

    print(memory.load_memory_variables({}))

    # ====================================================================================
    # LLM Chain作成
    # ====================================================================================
    # LLM Chain
    chain = LLMChain(
        llm=llm,             # LLMモデル
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

    # memoryオブジェクトから会話の履歴を取得して、変数historyに代入
    history = memory.chat_memory

    # 会話履歴をJSON形式の文字列に変換、整形して変数messagesに代入
    messages = json.dumps(messages_to_dict(
        history.messages), indent=2, ensure_ascii=False)

    instance.body = messages  # モデルのフィールドに値を代入します。
    print("savesavesave")
    print(messages)
    instance.save()  # データベースに保存します。

    return result


def make_questions(base_message):

    question_topick = [
        "- 相手がどのように感じているかを聞く",
        "- 地名がを含んだ内容の場合、土佐ではどうだったかを聞く",
        "- 薩長同盟との比較",
        "- 時代の変化などについて質問する",
        "- 龍馬とかかわり合いのあった人物を絡めて質問をする",
        "- 最も困難だと感じた出来事を絡めて質問する",
        "- 土佐藩における幼少期の体験を絡めて質問する",
        "- 明治維新の結果を絡めて質問する",
        "- あなたが最も尊敬する人物を交えて質問する",
        "- 土佐藩における生活を交えて質問する",
        "- 当時の社会の変化について質問する",
        "- 江戸時代と明治時代の違いについて質問する",
        "- 土佐藩時代の生活をについて質問する",
        "- 時代の変化をどのように捉えているか、を含んだ質問をする",
        "- 若き日のあなたに影響を与えた人物や出来事はを交えて質問する",
        "- 薩長同盟とあなたの活動の間で共通する点や相違点を含めて質問する",
        "- 当時の日本の社会や政治の変化に対するあなたの見解を交えて質問する",
        "- あなたの視点から見た江戸時代と明治時代の主な違いを聞く",
        "- あなたが直面した最大の挑戦を交えて質問する",
        "- 明治維新に至った過程や結果についてどのように思うかを考えて質問する",
        "- 吉田松陰を含めて質問文を考える",
        "- 高杉晋作を含めて質問文を考える",
        "- 西郷隆盛を含めて質問文を考える",
        "- 岩崎弥太郎を含めて質問文を考える",
        "- 勝海舟を含めて質問文を考える",
        "- 中岡慎太郎を含めて質問文を考える",
        "- お龍を含めて質問文を考える",
    ]
    selected_strings = random.sample(question_topick, 5)
    result_string = "\n".join(selected_strings)

    history = ""
    encoding: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    try:
        instance = BodyText.objects.get(id=1)  # 1はサンプルのIDです

        saved_memory = json.loads(instance.body)
        current_token = 0
        for i in range(len(saved_memory) - 2, -1, -2):
            human_item = saved_memory[i]
            ai_item = saved_memory[i + 1]
            # print('Human input:', human_item['data']['content'])
            temp_history = ""
            temp_history += "[AI]" + ai_item['data']['content'] + "\n"
            temp_history += "[Human]" + human_item['data']['content'] + "\n"
            tokens = encoding.encode(temp_history)
            tokens_count = len(tokens)
            print(f"{tokens_count=}")
            print(temp_history)
            if (2048 < tokens_count + current_token):
                break
            history = temp_history + history
            current_token += tokens_count

    except ObjectDoesNotExist:
        instance = BodyText()  # 該当のIDが存在しない場合、新たなインスタンスを作成します。
    except json.decoder.JSONDecodeError as e:
        # JSON文字列が正しくデコードできない場合の例外処理
        print(f"JSONDecodeError: {e}")

    result_string = history

    json_string = """
{
    "question" : ["質問文1","質問文2","質問文3"]
}
"""
    template = """
次のメッセージを主軸として、質問文を３つ作成してください。
坂本龍馬との会話を楽しむための返答分を3つ作成してください。
データのフォーマットはjsonデータ形式で返してください
フォーマット{json_string}

会話を行うための主軸にする話題は次のメッセージになります。
このメッセージを重視して回答を作成してください。
メッセージ:{base_message}

質問には以下の内容を含めながら行う
返答文はこれまでの会話を踏まえたものや、坂本龍馬のストーリーを踏まえたものを作成してください。
{result_string}
        """
    print(template)

    # プロンプトテンプレート
    prompt_template = PromptTemplate(
        input_variables=["json_string",
                         "base_message", "result_string"],  # 入力変数
        template=template,              # テンプレート
        validate_template=True,                  # 入力変数とテンプレートの検証有無
    )

    llm = ChatOpenAI(
        openai_api_base=os.getenv('OPENAI_API_BASE'),
        openai_api_key=os.getenv("OPENAI_API_KEY_AZURE"),
        model_name='gpt-4', model_kwargs={"deployment_id": "Azure-GPT-4-8k"}, temperature=0)

    chain = LLMChain(
        llm=llm,             # LLMモデル
        prompt=prompt_template,  # プロンプトテンプレート
        verbose=True,            # プロンプトを表示するか否か
        #        memory=memory,          # メモリ
    )
    # LLM Chain実行
    result = chain.predict(
        base_message=base_message,
        json_string=json_string,
        result_string=result_string)

    print("-----------質問文-----------")
    # questions = response["choices"][0]["message"]["content"]
    print(result)
    return result
    # 辞書をJSON形式の文字列に変換する
    # json_data = json.dumps(retdata,ensure_ascii=False)
