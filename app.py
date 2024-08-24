import os
import logging
from flask import Flask, request, jsonify
from modelscope import snapshot_download
from PIL import Image
from backend.rag import ragResults
from backend.voice import process_audio_files
from backend.picture import predict_deepfake_probability

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 向量模型下载
from modelscope import snapshot_download
model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5", cache_dir='.')

# 源大模型下载
from modelscope import snapshot_download
#model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')
model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='.')

app = Flask(__name__)


@app.route('/textMessage', methods=['POST'])
def getMessageInfo():
    try:
        data = request.json.get('data')  # 使用 JSON 请求体
        if not data:
            return jsonify({'status': 'error', 'message': 'Missing input data'}), 400

        vector_link = "AI-ModelScope/bge-small-zh-v1.5"
        llm_link = 'IEITYuan/Yuan2-2B-Mars-hf'

        embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
        document_dir = "./database"
        document_path = os.listdir(document_dir)[:10]
        model_path = './IEITYuan/Yuan2-2B-Mars-hf'

        '''
        if not os.path.exists(vector_link.split('/')[0]):
            vector_model_dir = snapshot_download(vector_link)
        if not os.path.exists(llm_link.split('/')[0]):
            llm_model_dir = snapshot_download(llm_link)
        '''

        respMessage = ragResults(embed_model_path, document_dir, document_path, model_path, data)
        return jsonify({'status': 'success', 'message': respMessage})

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/pictureMessage', methods=['POST'])
def getPictureInfo():
    try:
        # 从请求中获取图像文件
        image_file = request.files.get('image')  # 图像文件
        if not image_file:
            return jsonify({'status': 'error', 'message': '缺少图像文件'}), 400

        # 保存图像文件到临时目录
        temp_dir = './temp_images'
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, image_file.filename)

        # 保存上传的图像文件
        image_file.save(image_path)
        # 调试输出文件的类型和大小
        logger.info(f"Saved image to: {image_path}, Size: {os.path.getsize(image_path)} bytes")

        # 尝试打开图像以确保其有效性
        try:
            img = Image.open(image_path)
            img.verify()  # 验证图像的完整性
            logger.info("Image file is valid.")
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            return jsonify({'status': 'error', 'message': '无效的图像文件'}), 400

        # 预测图像的Deepfake概率
        probability = predict_deepfake_probability(image_path)

        # 清理临时文件
        os.remove(image_path)

        return jsonify({'status': 'success', 'message': probability})

    except Exception as e:
        logger.error(f"Error processing picture message: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/audioMessage', methods=['POST'])
def getAudioInfo():
    try:
        # 从请求中获取音频文件
        audio_file1 = request.files.get('audio1')  # 第一个音频文件
        audio_file2 = request.files.get('audio2')  # 第二个音频文件

        if not audio_file1 or not audio_file2:
            return jsonify({'status': 'error', 'message': '缺少音频文件'}), 400

        # 保存音频文件到临时目录
        temp_dir = './temp_audio'
        os.makedirs(temp_dir, exist_ok=True)
        audio_file_path1 = os.path.join(temp_dir, audio_file1.filename)
        audio_file_path2 = os.path.join(temp_dir, audio_file2.filename)

        # 保存上传的音频文件
        audio_file1.save(audio_file_path1)
        audio_file2.save(audio_file_path2)

        # 处理音频文件
        result = process_audio_files(audio_file_path1, audio_file_path2)
        print(result)
        # 清理临时文件
        os.remove(audio_file_path1)
        os.remove(audio_file_path2)

        return jsonify({'status': 'success', 'data': result['interpretation'] })

    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()
    logger.info("API service Stop running.")
