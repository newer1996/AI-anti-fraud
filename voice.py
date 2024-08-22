import numpy as np
from paddlespeech.cli.vector import VectorExecutor
from pydub import AudioSegment

def convert_to_wav(mp3_file_path, output_file_path):
    try:
        mp3_audio = AudioSegment.from_file(mp3_file_path)
        mp3_audio = mp3_audio.set_frame_rate(16000).set_sample_width(2)
        mp3_audio.export(output_file_path, format="wav")
        print(f"Converted {mp3_file_path} to {output_file_path}")
    except Exception as e:
        print(f"Error converting {mp3_file_path}: {e}")

def get_audio_embedding(audio_file_path):
    vec = VectorExecutor()
    try:
        result = vec(audio_file=audio_file_path, force_yes=True)
        return result
    except Exception as e:
        print(f"Error getting embedding for {audio_file_path}: {e}")
        return None

def compare_audio_embeddings(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return None  # 避免除以零的情况
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

def interpret_similarity_score(score):
    if score == 1:
        return "完全相同"
    elif 0.7 <= score < 1:
        return "高度相似"
    elif 0 <= score < 0.7:
        return "中度相似"
    elif score < 0:
        return "负相关"
    else:
        return "无相似性"
    

# 这里替换成前端传输的音频，根据app.py传入来定
audio_file_path1 = "./86.mp3"
audio_file_path2 = "./87.mp3"

# 转换音频文件到WAV格式
wav_file_path1 = "./audio1.wav"
wav_file_path2 = "./audio2.wav"

convert_to_wav(audio_file_path1, wav_file_path1)
convert_to_wav(audio_file_path2, wav_file_path2)

# 获取音频向量
embedding1 = get_audio_embedding(wav_file_path1)
embedding2 = get_audio_embedding(wav_file_path2)

# 比较两个音频向量
if embedding1 is not None and embedding2 is not None:
    similarity_score = compare_audio_embeddings(embedding1, embedding2)
    print(f"The similarity score between the two audios is: {similarity_score}")
    interpretation = interpret_similarity_score(similarity_score)
    print(f"Interpretation of similarity score: {interpretation}")
else:
    print("Failed to get embeddings for one or both audio files.")
