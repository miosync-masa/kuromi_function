import os
import streamlit as st
from dotenv import load_dotenv
import json
import requests
import openai
from openai import OpenAI
from openai import OpenAIError  # 修正
from openai_function_calling import Function, Parameter, JsonSchemaType
from openai_function_calling.tool_helpers import ToolHelpers
import logging
import pdfplumber
import random
from upstash_vector import Index

# 環境変数の読み込み
load_dotenv()

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieval用ディレクトリ
RETRIEVAL_DIR = "retrieval_data"

# OpenAIクライアントの初期化
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY が設定されていません。")
    st.error("APIキーが見つかりません。環境変数を確認してください。")

client = OpenAI(api_key=api_key)

# Streamlit ページ設定
st.set_page_config(page_title="Advanced Chat with Miosync", page_icon="💬", layout="wide")

# タイトルと説明
st.title("💬 Advanced Chat with Miosync")
st.write("Chat with an AI Partner. You can customize the model and system behavior.")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []  # 会話ログを空リストで初期化

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o-2024-08-06"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 8000

if "system_prompt" not in st.session_state:  # シンプルな文字列として初期化
    st.session_state.system_prompt = "You are a helpful assistant. Use tools intelligently."

# 履歴の長さ制限
MAX_HISTORY_LENGTH = 15

def trim_message_history():
    """
    履歴が長すぎる場合、古いメッセージを削除
    """
    if len(st.session_state.messages) > MAX_HISTORY_LENGTH:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY_LENGTH:]

# チャット履歴を表示するための Message Placeholder
message_placeholder = st.empty()

def update_chat_history():
    """
    チャット履歴を Message Placeholder に表示
    """
    with message_placeholder.container():
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='User-message' style='margin-bottom: 40px;'><b></b> {msg['content']}</div>", unsafe_allow_html=True)
            elif msg["role"] == "assistant":
                formatted_message = f"<b></b> {msg['content']}"
                st.markdown(
                    f"""
                    <div class="Partner-message" style="margin-bottom: 40px; padding: 20px; border-radius: 10px; background-color: #303030;">
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <img src="https://www.miosync.link/img/list-kuromi.png" 
                                width="50" height="50" 
                                style="border-radius: 50%;" />
                            <div style="flex-grow: 1; word-wrap: break-word; font-size: 16px; line-height: 1.6;">
                                {formatted_message}
                            </div>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

def generate_embedding(data: list[str], model: str = "text-embedding-ada-002") -> list[list[float]]:
    """
    テキストデータをベクターに変換する関数

    Args:
        data (list[str]): ベクター化するテキストデータのリスト。
        model (str): 使用するEmbeddingモデル。

    Returns:
        list[list[float]]: 各テキストのEmbeddingベクターのリスト。
    """
    try:
        if not data:
            logger.warning("generate_embeddingに空のデータリストが渡されました。")
            return []

        logger.debug(f"Generating embeddings for data: {data}")
        openai_result = client.embeddings.create(input=data, model=model)  # 'client' を使用
        logger.debug(f"OpenAI Embedding API Response: {openai_result}")

        result_vectors = []

        # OpenAIのレスポンスをベクター配列にマッピング
        for vector_response in openai_result.data:  # 'data' 属性にアクセス
            embedding = vector_response.embedding  # 'embedding' 属性にアクセス
            if embedding:
                result_vectors.append(embedding)
            else:
                logger.warning(f"Embeddingが存在しないデータ: {vector_response}")

        # レスポンスオブジェクトの内容をログに出力
        logger.debug(f"Generated embeddings: {result_vectors}")

        logger.info(f"Generated {len(result_vectors)} embeddings.")
        return result_vectors
    except openai.error.OpenAIError as e:  # OpenAIErrorを修正
        logger.exception(f"OpenAI API エラー: {e}")
        return []
    except Exception as e:
        logger.exception(f"ベクター生成中に予期せぬエラーが発生しました: {e}")
        return []


def search_documents(query: str, context: str = "", selected_category: str = "") -> dict:
    """
    クエリをエンベディングに変換し、ベクトル検索を実行し、
    結果に基づいてPDFを処理して内容を取得。

    Args:
        query (str): 検索クエリ。
        context (str): クエリの文脈情報。
        selected_category (str): カテゴリフィルタ。

    Returns:
        dict: 検索結果とPDF内容を含むレスポンス。
    """
    # クエリをエンベディングに変換
    embedding_list = generate_embedding([query])
    if not embedding_list:
        logger.error("ベクター生成に失敗しました。")
        return {"error": "ベクター生成に失敗しました。"}
    
    query_vector = embedding_list[0]
    
    # 環境変数からURLとトークンを取得
    endpoint = os.getenv("UPSTASH_VECTOR_ENDPOINT")
    token = os.getenv("UPSTASH_VECTOR_TOKEN")
    
    if not endpoint or not token:
        logger.error("UPSTASH_VECTOR_ENDPOINT または UPSTASH_VECTOR_TOKEN が設定されていません。")
        return {"error": "検索サービスの設定が正しくありません。"}
    
    index = Index(url=endpoint, token=token)
    
    try:
        # クエリの実行
        query_result = index.query(
            vector=query_vector,
            include_metadata=True,
            include_data=True,
            include_vectors=False,
            top_k=3,
        )

        # 検索結果の整形
        formatted_results = [
            {
                "id": result.id,
                "score": result.score,
                "metadata": result.metadata,
                "data": result.data
            }
            for result in query_result
        ]
        logger.info(f"検索結果: {len(formatted_results)} 件取得")

        # PDF処理の実行
        try:
            pdf_results = process_pdf_results(RETRIEVAL_DIR, formatted_results)

            # PDF内容を検索結果に追加
            for result in formatted_results:
                filename = result["metadata"].get("filename")
                if filename and filename in pdf_results and pdf_results[filename]:
                    result["pdf_content"] = pdf_results[filename][:500]  # 最初の500文字を追加
                else:
                    result["pdf_content"] = "関連する情報が見つかりませんでした。"

        except FileNotFoundError as e:
            logger.error(f"PDF処理エラー: {e}")
            return {"error": f"PDF処理中にエラーが発生しました: {str(e)}"}

        return {"results": formatted_results}

    except Exception as e:
        logger.exception("Upstash Vectorへのリクエスト中にエラーが発生しました。")
        return {"error": f"検索中にエラーが発生しました: {str(e)}"}


def process_pdf_results(retrieval_dir, formatted_results):
    """
    クエリ結果に基づいてPDFを処理し、内容を抽出する。

    Args:
        retrieval_dir (str): PDFが格納されているディレクトリ。
        formatted_results (list[dict]): 整形されたクエリ結果。

    Returns:
        dict: ファイル名と抽出された内容の辞書。
    """
    if not os.path.exists(retrieval_dir):
        raise FileNotFoundError(f"ディレクトリが存在しません: {retrieval_dir}")

    processed_results = {}
    for result in formatted_results:
        filename = result["metadata"].get("filename")
        if not filename:
            continue  # ファイル名がない場合はスキップ

        file_path = os.path.join(retrieval_dir, filename)

        if os.path.exists(file_path):
            # PDFを開いて内容を抽出
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            processed_results[filename] = text
        else:
            processed_results[filename] = None  # ファイルが見つからない場合

    return processed_results
                
# サイドバー設定
st.sidebar.title("Settings")
st.sidebar.subheader("Model Configuration")
st.session_state.selected_model = st.sidebar.selectbox(
    "Choose a model:",
     ["gpt-4o-2024-08-06", "gpt-4o-mini", "chatgpt-4o-latest"],
    index=["gpt-4o-2024-08-06", "gpt-4o-mini", "chatgpt-4o-latest"].index(st.session_state.selected_model)
)

# サイドバーでSystem Promptを設定
st.sidebar.subheader("System Prompt")
st.session_state.system_prompt = st.sidebar.text_area(
    "Set the system's behavior prompt:",
    st.session_state.system_prompt
)

# サイドバーで温度、Top-p、Max tokens を調整
st.sidebar.subheader("Generation Parameters")
st.session_state.temperature = st.sidebar.slider(
    "Temperature (Creativity)", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.temperature, 
    step=0.01
)
st.session_state.top_p = st.sidebar.slider(
    "Top-p (Sampling)", 
    min_value=0.0, 
    max_value=1.0, 
    value=st.session_state.top_p, 
    step=0.01
)
st.session_state.max_tokens = st.sidebar.slider(
    "Max Tokens (Response Length)", 
    min_value=100, 
    max_value=12096, 
    value=st.session_state.max_tokens, 
    step=100
)

# search_documentsをOpenAI Functionとして定義
search_documents_function = Function(
    name="search_documents",
    description="Search for relevant documents using Upstash Vector DB.",
    parameters=[
        Parameter(
            name="query",
            type=JsonSchemaType.STRING,
            description="The search query."
        ),
        Parameter(
            name="context",
            type=JsonSchemaType.STRING,
            description="Context for the query."
        ),
        Parameter(
            name="selected_category",
            type=JsonSchemaType.STRING,
            description="The category to filter documents."
        )
    ]
)

tools = ToolHelpers.from_functions([search_documents_function])

def generate_response(user_input):
    """
    OpenAI APIを使用して応答を生成
    モデルがfunction callingすればそれを実行、その結果を再度モデルに渡し、最終回答を取得
    """
    try:
        # ユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        trim_message_history()
        update_chat_history()

        response = client.chat.completions.create(
            model=st.session_state.selected_model,
            messages=[
                {"role": "system", "content": st.session_state.system_prompt}  # 動的に追加
            ] + st.session_state.messages,  # 会話履歴を追加
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            max_tokens=st.session_state.max_tokens,
            tools=tools,  # ツールを追加
            tool_choice="auto"  # 自動選択
        )

        response_message = response.choices[0].message

        # tool_callが発生した場合
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            logger.info(f"Function Called: {function_name}")
            logger.info(f"Arguments: {arguments}")

            if function_name == "search_documents":
                query = arguments.get("query", "")

                # 検索クエリを実行
                function_response = search_documents(query)

                if "error" in function_response:
                    # エラーメッセージを表示
                    error_message = function_response["error"]
                    st.error(error_message)
                else:
                    # 検索結果を整形してPDF処理を実行
                    formatted_results = [
                        {
                            "id": result["id"],
                            "score": result["score"],
                            "metadata": result["metadata"],
                            "data": result.get("data")
                        }
                        for result in function_response.get("results", [])
                    ]

                    try:
                        # PDF処理を実行
                        pdf_results = process_pdf_results(RETRIEVAL_DIR, formatted_results)

                        # PDF内容をプロンプトに統合
                        pdf_summary = []
                        for filename, content in pdf_results.items():
                            if content:
                                pdf_summary.append(f"ファイル名: {filename}\n内容:\n{content[:500]}...\n")
                            else:
                                pdf_summary.append(f"ファイル名: {filename}\n内容: 関連する情報が見つかりませんでした。\n")

                        # プロンプトにPDF結果を統合
                        prompt_context = "\n".join(pdf_summary)

                        # モデルへの再問い合わせ
                        final_response = client.chat.completions.create(
                            model=st.session_state.selected_model,
                            messages=[
                                {"role": "system", "content": st.session_state.system_prompt},  # 動的プロンプト
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": prompt_context},  # PDF結果をプロンプトに統合
                            ],
                            temperature=st.session_state.temperature,
                            top_p=st.session_state.top_p,
                            max_tokens=st.session_state.max_tokens
                        )

                        # モデルの最終応答をチャット履歴に追加
                        final_msg = final_response.choices[0].message
                        if hasattr(final_msg, 'content') and final_msg.content:
                            st.session_state.messages.append({"role": "assistant", "content": final_msg.content})
                        else:
                            logger.error("Final message content is missing.")
                    
                    except Exception as e:
                        logger.exception("Error during PDF processing.")
                        st.error(f"An error occurred during PDF processing: {e}")
        else:
            # function callなしで直接回答取得
            if hasattr(response_message, 'content') and response_message.content:
                st.session_state.messages.append({"role": "assistant", "content": response_message.content})

        update_chat_history()

    except Exception as e:
        logger.exception("Error during API call.")
        st.error(f"An error occurred: {e}")

# 入力フォーム(下部)
with st.form(key="chat_form"):
    user_input = st.text_input("Your message:", "")
    submit_button = st.form_submit_button("Send")

if submit_button:
    if user_input and user_input.strip():
        generate_response(user_input)
    else:
        st.error("Please enter a valid message.")

if st.button("Reset Chat"):
    # 履歴を完全に空リストで初期化
    st.session_state.messages = []  
    st.success("Chat history reset.")
    update_chat_history()
else:
    update_chat_history()