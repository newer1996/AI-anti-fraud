import os.path

from flask import Flask, request
from modelscope import snapshot_download
from backend.rag import ragResults

app = Flask(__name__)

@app.route('/textMessage', methods=['POST'])
def getMessageInfo(): #传递文字信息
    data = request.form['info']
    vector_link = "AI-ModelScope/bge-small-zh-v1.5" #向量模型链接
    llm_link = 'IEITYuan/Yuan2-2B-Mars-hf' #源大模型链接

    embed_model_path = './AI-ModelScope/bge-small-zh-v1___5'
    doecment_path = './knowledge.txt'
    model_path = './IEITYuan/Yuan2-2B-Mars-hf'

    if not os.path.exists(vector_link.split('/')[0]):
        vector_model_dir = snapshot_download(vector_link)
    if not os.path.exists(llm_link.split('/')[0]):
        llm_model_dir = snapshot_download(llm_link)

    #ragRes = ragResults(embed_model_path, doecment_path)
    respMessage = ragResults(embed_model_path, doecment_path, model_path, data)
    data_json = {}
    data_json['status'] = 'success'
    data_json['message'] = respMessage
    return data_json

@app.route('/pictureMessage')
def getPictureInfo():
    return ''

@app.route('/audioMessage')
def getAudioInfo():
    return ''

@app.route("/")
def printHello():


    return "Hello world"


if __name__ == '__main__':
    app.run()
