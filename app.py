from flask import Flask, request

app = Flask(__name__)

@app.route('/textMessage', methods=['POST'])
def getMessageInfo(): #传递文字信息
    data = request.form['info']
    '''
    大模型的相关接口
    '''
    respMessage = ''
    return respMessage

@app.route('/pictureMessage')
def getPictureInfo():
    return ''

@app.route('/audioMessage')
def getAudioInfo():
    return ''


if __name__ == '__main__':
    app.run()
