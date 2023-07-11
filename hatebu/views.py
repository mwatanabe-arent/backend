from email.quoprimime import unquote
from urllib.parse import unquote
import json
import random
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from bs4 import BeautifulSoup # スクレイピングするため（備考欄の取得のみ）
import requests

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response

from langchain import PromptTemplate
from llama_index import LLMPredictor, ServiceContext, PromptHelper
from langchain import OpenAI
from llama_index import GPTVectorStoreIndex 
from llama_index.readers import BeautifulSoupWebReader

from common.utils import make_questions

class Topics(APIView):
    def get(self, request):

        hatebu_top_json = self.get_hatebu_top_json(3)
        #print(hatebu_top_json)
        #return JsonResponse(hatebu_top_json , safe=False)

        return HttpResponse(hatebu_top_json, content_type="application/json")

    def get_hatebu_top_json(self,count):
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
                category= a_tag['data-entry-category'].replace('\n', '')

                entry = {
                    "title":text,
                    "link":link,
                    "category":category,
                }

                # a_tagの中の要素を配列に入れ直す
                temp_a_tag.append(entry)

        if count <= len(temp_a_tag) :
            a_tag_list = random.sample(temp_a_tag, count)
        else:
            # 要素数が3未満の場合の処理
            # 例えば、すべての要素を抽出するなどの操作を行う
            a_tag_list = temp_a_tag

        print("json_data")

        # データをJSON形式に変換
        json_data = json.dumps(a_tag_list, ensure_ascii=False, indent=4)

        # JSONデータを表示
        #print(json_data)

        return json_data


class URL(APIView):
    def get(self, request):
        
        request_url = request.GET.get('url')
        # unquote関数でURLをデコード
        decoded_url = unquote(request_url, encoding='utf-8')

        print(decoded_url)

        #web_text = self.get_webpage_texts(decoded_url)
        #print(web_text)

        web_summary = self.web_summary(decoded_url)
        questions = make_questions(web_summary.response)
        print("web_summary")
        print(web_summary)
        print("dir(web_summary)")
        print(dir(web_summary))
        print("web_summary.response")
        print(web_summary.response)
        print("questions")
        print(questions)

        # web_summaryをJsonの文字列に渡すために変換
        #web_summary = json.dumps(web_summary, ensure_ascii=False, indent=4)
        web_summary = web_summary.response
        #web_summary = web_summary.encode('ascii', 'ignore').decode('ascii')

        retdata = {
            "message":web_summary,
            "question_json":questions
        }
        
        #print(retdata)
        json_data = json.dumps(retdata)
        print("json_data")
        print(json_data)

        return Response(retdata)
        #return Response(json_data)
        #return Response(web_summary)

    # Webページからテキストを抽出
    def get_webpage_texts(self,url):
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
    
    def web_summary(self,url):

        template = """ このデータの要約を{num}個作成してください。
        それぞれの要約は日本語で100文字以上200文字以内で作成してください。　
        また、要約には{action}を入れてください。　
        """

        prompt = PromptTemplate(
            input_variables=["num", "action"],
            template = template,
        )


        # define LLM
        max_input_size = 4096
        num_output = 2048  #2048に拡大
        max_chunk_overlap = 0.2
        prompt_helper = PromptHelper(max_input_size, num_output)

        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=2048))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)


        #documents = BeautifulSoupWebReader().load_data(urls=["https://ja.wikipedia.org/wiki/%E5%A4%A7%E8%B0%B7%E7%BF%94%E5%B9%B3"])
        documents = BeautifulSoupWebReader().load_data(urls=[url])
        index = GPTVectorStoreIndex.from_documents(documents,  service_context=service_context) 
        query_engine = index.as_query_engine(service_context=service_context)

        answer = query_engine.query(prompt.format(num=3, action="楽しい内容"))
        #print(answer)
        return answer

