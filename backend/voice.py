import numpy as np
from paddlespeech.cli.vector import VectorExecutor
from pydub import AudioSegment

def convert_to_wav(mp3_file_path, output_file_path):
    """Convert an MP3 file to WAV format."""
    try:
        mp3_audio = AudioSegment.from_file(mp3_file_path)
        mp3_audio = mp3_audio.set_frame_rate(16000).set_sample_width(2)
        mp3_audio.export(output_file_path, format="wav")
        print(f"Converted {mp3_file_path} to {output_file_path}")
    except Exception as e:
        print(f"Error converting {mp3_file_path}: {e}")

def get_audio_embedding(audio_file_path):
    """Get the audio embedding for the given audio file."""
    vec = VectorExecutor()
    try:
        result = vec(audio_file=audio_file_path, force_yes=True)
        return result
    except Exception as e:
        print(f"Error getting embedding for {audio_file_path}: {e}")
        return None

def compare_audio_embeddings(embedding1, embedding2):
    """Compare two audio embeddings and return the cosine similarity score."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return None  # Avoid division by zero
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

def interpret_similarity_score(score):
    """Interpret the cosine similarity score."""
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

def process_audio_files(audio_file_path1, audio_file_path2):
    """
    Process two audio files, convert them to WAV, extract embeddings,
    compare them, and return the similarity score and interpretation.
    
    Parameters:
        audio_file_path1 (str): Path to the first audio file.
        audio_file_path2 (str): Path to the second audio file.
    
    Returns:
        dict: A dictionary containing the similarity score and interpretation.
    """
    # Convert audio files to WAV format
    wav_file_path1 = audio_file_path1.replace('.mp3', '.wav')
    wav_file_path2 = audio_file_path2.replace('.mp3', '.wav')
    
    convert_to_wav(audio_file_path1, wav_file_path1)
    convert_to_wav(audio_file_path2, wav_file_path2)

    # Get audio embeddings
    embedding1 = get_audio_embedding(wav_file_path1)
    embedding2 = get_audio_embedding(wav_file_path2)

    result = {}
    if embedding1 is not None and embedding2 is not None:
        similarity_score = compare_audio_embeddings(embedding1, embedding2)
        result['similarity_score'] = similarity_score
        result['interpretation'] = interpret_similarity_score(similarity_score)
    else:
        result['error'] = "未能成功处理声音，请确认声音文件格式正确！"

    return result

# Example usage (this part should be removed or commented out in production)
# audio_file_path1 = "./86.mp3"
# audio_file_path2 = "./87.mp3"
# result = process_audio_files(audio_file_path1, audio_file_path2)
# print(result)
