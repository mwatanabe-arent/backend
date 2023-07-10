# ベースとなるDockerイメージを指定
FROM python:3

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係のリストをコピー
COPY requirements.txt .

# Python依存関係のインストール
RUN pip install -r requirements.txt

# Djangoプロジェクトをコピー
COPY . .

# ポートを公開
EXPOSE 8000

# Djangoサーバーを起動
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
