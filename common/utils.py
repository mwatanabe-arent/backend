

from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import os
import requests
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


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


def make_questions(base_message):

    json_string = """
{
    "question" : ["質問文1","質問文2","質問文3"]
}
"""
    sform = f"""
        次のメッセージから質問文を３つ作成してください。
        データのフォーマットはjsonデータ形式で返してください
        フォーマット{json_string}

        メッセージ:{base_message}
        """
    print(sform)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{sform}"}],
    )
    print("-----------質問文-----------")
    questions = response["choices"][0]["message"]["content"]
    print(questions)
    return questions
    # 辞書をJSON形式の文字列に変換する
    # json_data = json.dumps(retdata,ensure_ascii=False)
