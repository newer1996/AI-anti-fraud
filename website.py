import streamlit as st
from PIL import Image
import requests  # 导入requests库用于与后端API交互

# 定义聊天记录列表
chat_history = []

def call_backend_service(input_data, input_type):
    url = ""
    if input_type == "text":
        url = "http://127.0.0.1:5000/textMessage"
        response = requests.post(url, json={"data": input_data})
    elif input_type == "image":
        url = "http://127.0.0.1:5000/pictureMessage"
        file = input_data['image']
        # 读取文件内容并准备为请求所需格式
        files = {'image': (file.name, file)}
        file.seek(0)
        response = requests.post(url, files=files)
    elif input_type == "audio":
        url = "http://127.0.0.1:5000/audioMessage"  # 修正 URL
        files = {'audio1': input_data[0], 'audio2': input_data[1]}
        response = requests.post(url, files=files)
        
    if response.status_code == 200:
        return response.json()  # 返回 JSON 数据
    else:
        return f"错误: {response.json().get('message', '无法处理请求')}"

# 设置页面标题和布局
st.set_page_config(page_title="AI反诈助手", layout="wide")

# 假设的数据处理函数
def get_random_greetings():
    greetings = "你好！欢迎使用AI反诈助手。"
    st.sidebar.markdown(greetings)
    return greetings

# 登录函数
def login():
    st.sidebar.header("🔒 登录")
    username = st.sidebar.text_input("用户名", "admin")
    password = st.sidebar.text_input("密码", "admin", type="password")
    
    if st.sidebar.button("登录"):
        if username == "admin" and password == "admin":
            st.session_state['logged_in'] = True
            st.session_state['user_name'] = username
            st.sidebar.success("登录成功！")
            st.rerun()
        else:
            st.sidebar.error("用户名或密码错误，请重试。")

# 检查用户是否登录
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    login()
else:
    # 用户已登录，显示主应用内容
    st.title("🛡️ AI反诈助手")
    
    # 显示用户名和默认信息
    st.sidebar.header("👤 用户信息")
    st.sidebar.markdown(f"欢迎, {st.session_state['user_name']}!")

    # 用户信息输入区域
    user_info = st.sidebar.text_area("请输入其他信息", "")

    # 选择输入方式
    option = st.sidebar.radio("📡 选择输入方式", ("文本对话", "图像输入", "音频输入"))

    # 文本对话部分
    if option == "文本对话": 
        user_input = st.text_input("输入您的消息", key="text_input")
        send_button = st.button("发送")
        process_status = st.empty()
        if send_button and user_input:
            process_status.write("处理中...")
            response = call_backend_service(user_input, "text")
            
            # 检查响应是否为字典并包含'status'和'message'
            if isinstance(response, dict):
                if response.get('status') == 'success' and 'message' in response:
                    message = response['message']
                    process_status.write(message)
                    chat_history.append(f"您: {user_input}")
                    chat_history.append(f"助手: {message}")
                else:
                    # 如果返回的状态不是'success'，处理错误
                    process_status.write(f"处理错误: {response.get('message', '未知错误')}")
                    chat_history.append(f"您: {user_input}")
                    chat_history.append(f"助手: {response.get('message', '未知错误')}")
            else:
                # 如果响应不是字典，显示错误信息
                process_status.write("处理错误: 无法获取助手的回复")
                chat_history.append(f"您: {user_input}")
                chat_history.append("助手: 返回的数据格式不正确")

        # 聊天记录展示区域
        st.markdown("### 聊天记录")
        for chat in chat_history:
            st.write(chat)

    # 图像输入部分
    elif option == "图像输入":
        uploaded_image = st.file_uploader("上传您的图像", type=["jpg", "jpeg", "png"])
        process_status = st.empty()  # 创建占位符用于显示处理状态
        if uploaded_image is not None:
            # 将上传的文件转换为字典形式，以便传递给 call_backend_service
            files = {'image': uploaded_image}
            send_button = st.button("发送")
            backend_response = st.empty()  # 用于显示后端返回的结果
            if send_button:
                process_status.write("处理中...")
                # 直接调用后端服务
                response = call_backend_service(files, "image")
                # 更新处理状态
                process_status.write("处理完成！")
                backend_response.text_area("分析结果：", response['message'], height=200)
    # 音频输入部分
    elif option == "音频输入":
        st.subheader("🎙️ 音频输入")
        uploaded_audio_1 = st.file_uploader("上传第一个音频", type=["mp3", "wav"], key="audio1")
        uploaded_audio_2 = st.file_uploader("上传第二个音频", type=["mp3", "wav"], key="audio2")
        send_button = st.button("发送")

        process_status_1 = st.empty()
        process_status_2 = st.empty()
        backend_response = st.empty()  # 用于显示后端返回的结果

        if send_button:
            if uploaded_audio_1 is not None and uploaded_audio_2 is not None:
                process_status_1.write("处理中...")
                # 调用后端服务，传递两个音频文件
                response_audio = call_backend_service((uploaded_audio_1, uploaded_audio_2), "audio")
                process_status_1.write("处理完成！")
                backend_response.text_area("分析结果", response_audio['data'], height=100)

# 显示随机问候语
get_random_greetings()
