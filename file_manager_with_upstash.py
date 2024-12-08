import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from upstash_vector import Index
from llama_parse import LlamaParse  # LlamaParseライブラリを使用
from uuid import uuid4

# 環境変数の読み込み
load_dotenv()
api_key = os.getenv("LLAMA_PARSE_API_KEY")
upstash_vector_endpoint = os.getenv("UPSTASH_VECTOR_ENDPOINT")
upstash_vector_token = os.getenv("UPSTASH_VECTOR_TOKEN")
upstash_vector_dimension = int(os.getenv("UPSTASH_VECTOR_DIMENSION", 1536))
client = OpenAI(api_key=os.getenv("OPENAI_EMBEDDING_API_KEY"))

# Streamlitページ設定
st.set_page_config(page_title="Advanced File Management", page_icon="📄", layout="wide")

# サポートされるファイル形式
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".doc", ".docx", ".xlsx", ".csv", ".json"]

# Upstash Vector DBの設定
vector_index = Index(url=upstash_vector_endpoint, token=upstash_vector_token)

# LlamaParseの設定
llama_parser = LlamaParse(
    api_key=api_key,  # APIキーを指定
    result_type="markdown",
    verbose=True
)

# サイドバーでアップロード
st.sidebar.title("Admin Panel")
uploaded_file = st.sidebar.file_uploader("Upload a file for indexing", type=["txt", "pdf", "csv", "json"])

# データセットカテゴリの選択
category = st.sidebar.selectbox(
    "Choose a dataset category",
    [
        "🐱 Sanrio: キャラクターやプロジェクト資料",
        "🧪 Amadeus: 研究や技術資料",
        "📂 Other: その他のデータ"
    ]
)

# カテゴリのキーをシンプルにするための処理
category_key = category.split(":")[0].strip("🐱🧪📂 ")

# ファイルのアップロード処理
if uploaded_file and st.sidebar.button("Upload and Index"):
    try:
        # ファイル内容を読み込み
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()

        # ファイル形式のバリデーション
        if file_extension not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types are: {', '.join(SUPPORTED_FILE_TYPES)}")

        file_contents = uploaded_file.read()
        if not file_contents:
            raise ValueError("The uploaded file is empty. Please upload a valid file.")

        # アップロード成功メッセージ
        st.success("File uploaded successfully!")

        # ファイルのメタ情報を設定
        extra_info = {"file_name": file_name}

        # LlamaParse で解析
        documents = llama_parser.load_data(file_contents, extra_info=extra_info)
        if not documents or len(documents) == 0:
            raise ValueError("No documents were returned by LlamaParse. Ensure the file is valid.")

        # 解析結果からテキストを取得
        parsed_text = documents[0].text  # 最初のドキュメントのテキストを取得
        if not parsed_text.strip():
            raise ValueError("The parsed document is empty. Verify the file content.")

        # 解析結果をプレビュー
        st.write("Parsed Text Preview:", parsed_text[:500])

        # OpenAI で埋め込み生成
        try:
            response = client.embeddings.create(input=parsed_text,
            model="text-embedding-ada-002")

            # エンベディングを取得
            embedding = response.data[0].embedding
            st.write("Embedding created successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}")

        # ベクトルデータをUpstashに保存
        file_id = str(uuid4())  # 一意のIDを生成
        metadata = {"filename": file_name, "category": category}
        vector_index.upsert(
            vectors=[
                (file_id, embedding, metadata)
            ]
        )
        st.success(f"File '{file_name}' indexed under category '{category}'.")
        
        # ページをリフレッシュ
        st.rerun()

    except ValueError as ve:
        # ファイルや解析の検証エラーを表示
        st.error(f"Validation error: {ve}")
    except Exception as e:
        # 一般的なエラーを表示
        st.error(f"An error occurred during parsing or indexing: {e}")

# アップロード済みファイルの一覧表示
st.title("Uploaded Files")
st.write("Below is a list of files currently indexed:")

# インデックスのメタ情報を表示
st.title("Index Information")
try:
    # インデックス情報を取得
    index_info = vector_index.info()

    # 情報を表示
    st.write(f"**Vector Count:** {index_info.vector_count}")
    st.write(f"**Pending Vector Count:** {index_info.pending_vector_count}")
    st.write(f"**Index Size:** {index_info.index_size}")
    st.write(f"**Dimension:** {index_info.dimension}")
    st.write(f"**Similarity Function:** {index_info.similarity_function}")

except Exception as e:
    st.error(f"Error retrieving index information: {e}")
