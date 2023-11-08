import openai
import streamlit as st

'''with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
'''
st.title("💬 藏格矿业知识问答")
st.caption("回答关于藏格矿业各种规章制度的问题")

def chat():
    messages = request.form.get("prompts", None)
    prompts = json.loads(messages)

    path = './vector_store'
    data = pd.read_csv('./wenben.csv', encoding='gbk')
    if os.path.exists('./vector_store'):
        faiss = FAISS.FAISS(data=data)
        faiss.vector_read()
    else:
        faiss = None

    if not faiss:
        text_list = data.paragraph.values
        vector_matrix = FAISS.Embedding().get_embedding(text_list)
        faiss = FAISS.FAISS(vector_matrix=vector_matrix, data=data)
        faiss.vector_store()
    query_vector = FAISS.Embedding().get_embedding(prompts[0]['content'])[0]
    results = faiss.vector_search(query_vector, top_k=4)
    content = ''
    for txt in results:
        content += txt
    messages1 = [
        {"role": "user", "content": f"```{content}```\n<{prompts[0]['content']}>"},
        {'role': 'system',
         'content': "你是一名企业管理者，你正在回答新人对公司规章的提问，你精准专业的回答对公司的健康发展起着至关重要的作用。请确保使用中文回答。公司为你提供的相关信息由三个反引号```包裹，例如```相关信息```。新人的问题由<>包裹，例如<问题>。你需要说明答案来自哪个文件的哪一章哪一条"}
    ]

#Streamlit 原版模版
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！你可以向我咨询企业规章制度"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    openai.api_base = 'https://api.openai-proxy.org/v1'
    openai.api_key = 'sk-yE9AUncq99czee9GsytQYP4QxYuZal44kPMx5y9rkqojGVu3'
    st.session_state.messages.append(messages1)
    st.chat_message("user").write(prompt)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
