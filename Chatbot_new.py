import openai
import streamlit as st
import requests
import json

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

# 初始化Streamlit应用
st.title("💬 藏格矿业知识问答")
st.caption("回答关于藏格矿业各种规章制度的问题")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# 加载向量库文件
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
new_db = FAISS.load_local('./', embeddings)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # 设置OpenAI API密钥
    openai.api_base = 'https://api.openai-proxy.org/v1'
    openai.api_key = 'sk-yE9AUncq99czee9GsytQYP4QxYuZal44kPMx5y9rkqojGVu3'

    # 用户输入消息
    user_message = {"role": "user", "content": prompt}

    # 获取匹配的向量文档
    docs = new_db.similarity_search(prompt)
    content = ''.join(doc['page_content'] for doc in docs)

    # 构建对话消息
    system_message_prompt = SystemMessagePromptTemplate.from_template(content)
    human_template = "```{docs}```\n<{question}>"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    msg = chat_prompt.format_prompt(docs=content, question=prompt)

    st.session_state.messages.append(user_message)
    st.chat_message("user").write(prompt)

    # 使用OpenAI GPT-3生成回复
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    assistant_msg = response.choices[0].message
    st.session_state.messages.append(assistant_msg)
    st.chat_message("assistant").write(assistant_msg.content)
