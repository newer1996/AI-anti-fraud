import streamlit as st
from PIL import Image
import requests  # 导入requests库用于与后端API交互

# 定义聊天记录列表
chat_history = []

# 与后端连接的函数
def call_backend_service(input_data, input_type):
    """
    调用后端服务，根据输入类型选择不同的接口。
    """
    url = "http://127.0.0.1:5000"
    if input_type == "text":
        url = "http://backend-service/api/text"  # 文本处理的后端接口
        response = requests.post(url, json={"data": input_data})
    elif input_type == "image":
        url = "http://backend-service/api/image"  # 图像处理的后端接口
        files = {'file': input_data}  # 直接发送文件
        response = requests.post(url, files=files)
    elif input_type == "audio":
        url = "http://backend-service/api/audio"  # 音频处理的后端接口
        files = {'audio1': input_data[0], 'audio2': input_data[1]}  # 发送两个音频文件
        response = requests.post(url, files=files)

    return response.text if response.status_code == 200 else "错误: 无法处理请求"

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
        process_status = st.empty()  # 创建占位符用于显示处理状态
        if send_button and user_input:
            process_status.write("处理中...")
            # 调用后端服务
            response = call_backend_service(user_input, "text")
            # 更新处理状态
            process_status.write(f"{response}")
            chat_history.append(f"您: {user_input}")
            chat_history.append(f"助手: {response}")

        # 聊天记录展示区域
        st.markdown("### 聊天记录")
        for chat in chat_history:
            st.write(chat)

    # 图像输入部分
    elif option == "图像输入":
        uploaded_image = st.file_uploader("上传您的图像", type=["jpg", "jpeg", "png"])
        process_status = st.empty()  # 创建占位符用于显示处理状态
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption='上传的图像', use_column_width=True)
            send_button = st.button("发送")
            backend_response = st.empty()  # 用于显示后端返回的结果
            if send_button:
                process_status.write("处理中...")
                # 直接调用后端服务
                response = call_backend_service(uploaded_image, "image")
                # 更新处理状态
                process_status.write(f"处理完成！")
                backend_response.text_area("后端返回的结果", response, height=100)

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
                backend_response.text_area("分析结果", response_audio, height=100)

# 显示随机问候语
get_random_greetings()
