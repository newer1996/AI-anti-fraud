import os
import logging
from flask import Flask, request, jsonify
from modelscope import snapshot_download
from backend.rag import ragResults

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        document_path = ['./knowledge.txt']
        model_path = './IEITYuan/Yuan2-2B-Mars-hf'

        if not os.path.exists(vector_link.split('/')[0]):
            vector_model_dir = snapshot_download(vector_link)
        if not os.path.exists(llm_link.split('/')[0]):
            llm_model_dir = snapshot_download(llm_link)

        respMessage = ragResults(embed_model_path, document_path, model_path, data)
        return jsonify({'status': 'success', 'message': respMessage})

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/pictureMessage', methods=['POST'])
def getPictureInfo():
    return jsonify({'status': 'success', 'message': 'Picture message received'})

@app.route('/audioMessage', methods=['POST'])
def getAudioInfo():
    return jsonify({'status': 'success', 'message': 'Audio message received'})

@app.route("/")
def printHello():
    return "Hello world"

if __name__ == '__main__':
    app.run()
    logger.info("API service is running.")
