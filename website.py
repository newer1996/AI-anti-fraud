import streamlit as st
from PIL import Image

# 设置页面标题
st.set_page_config(page_title="AI反诈助手", layout="wide")

# 登录部分
def login():
    st.sidebar.header("登录")
    username = st.sidebar.text_input("用户名", "")
    password = st.sidebar.text_input("密码", "", type="password")
    
    if st.sidebar.button("登录"):
        if username == "admin" and password == "admin":
            st.session_state['logged_in'] = True
            st.session_state['user_name'] = username
            st.success("登录成功！")
            # 直接跳转到主页面
            st.experimental_rerun()  # 重新运行应用程序
        else:
            st.error("用户名或密码错误，请重试。")

# 如果用户没有登录，显示登录界面
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    login()
else:
    # 用户已登录，显示主应用内容
    st.title("AI反诈助手")
    
    # 显示用户名和默认信息
    st.sidebar.header("用户信息")
    st.sidebar.write(f"欢迎, {st.session_state['user_name']}!")
    
    user_info = st.sidebar.text_area("请输入其他信息", "")
    
    # 将选择输入方式的按钮放置在侧边栏
    option = st.sidebar.radio("选择输入方式", ("文本对话", "图像输入", "音频输入"), key="input_option")
    
    if option == "文本对话":
        st.sidebar.header("大模型参数调节")
        param1 = st.sidebar.slider("参数1", min_value=0.0, max_value=1.0, value=0.5)
        param2 = st.sidebar.slider("参数2", min_value=0, max_value=100, value=50)

    # 创建一个用于展示聊天记录的区域
    chat_history_container = st.empty()
    chat_history = []

    # 创建一个下方的输入区域
    st.subheader("聊天记录")
    with st.container():
        # 聊天记录展示区域
        for chat in chat_history:
            st.write(chat)

        # 文本对话
        user_input = ""
        if option == "文本对话":
            user_input = st.text_input("输入您的消息", "", key="text_input")

        # 图像输入
        elif option == "图像输入":
            uploaded_image = st.file_uploader("上传您的图像", type=["jpg", "jpeg", "png"], key="image_uploader")
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption='上传的图像', use_column_width=True)

        # 音频输入
        elif option == "音频输入":
            st.subheader("音频输入")
            uploaded_audio_1 = st.file_uploader("上传第一个音频", type=["mp3", "wav"], key="audio_uploader_1")
            uploaded_audio_2 = st.file_uploader("上传第二个音频", type=["mp3", "wav"], key="audio_uploader_2")

            if uploaded_audio_1 is not None:
                st.audio(uploaded_audio_1, format='audio/wav')

            if uploaded_audio_2 is not None:
                st.audio(uploaded_audio_2, format='audio/wav')

        # 统一发送按钮
        if st.button("发送"):
            if option == "文本对话" and user_input:
                response = "模型的反馈: " + user_input  # 这里替换为实际的API调用
                chat_history.append(f"您: {user_input}")
                chat_history.append(f"助手: {response}")
            elif option == "图像输入" and uploaded_image is not None:
                response = "图像已处理"  # 这里替换为实际的API调用
                chat_history.append("您: [图像上传]")
                chat_history.append(f"助手: {response}")
            elif option == "音频输入" and (uploaded_audio_1 or uploaded_audio_2):
                response = "音频已处理，声纹对比结果：一致"  # 这里替换为实际的API调用
                chat_history.append("您: [音频上传]")
                chat_history.append(f"助手: {response}")

    # 更新聊天记录容器
    with chat_history_container:
        for chat in chat_history:
            st.write(chat)
