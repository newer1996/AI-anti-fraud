import os
import pickle
from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

class EmbeddingModel:
    def __init__(self, path: str, device: str = 'cuda') -> None:
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device)
        print(f'Loading EmbeddingModel from {path} on {self.device}.')

    def get_embeddings(self, texts: List) -> List[float]:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()

class VectorStoreIndex:
    def __init__(self, document_dir: str, document_path: List, embed_model: EmbeddingModel, index_file: str = 'vector_index.pkl') -> None:
        self.index_file = index_file
        self.documents = []

        # 检查是否存在向量库
        if os.path.exists(self.index_file):
            print(f'Loading existing vector index from {self.index_file}.')
            with open(self.index_file, 'rb') as f:
                self.vectors, self.documents = pickle.load(f)
            self.embed_model = embed_model  # 确保 embed_model 被初始化
        else:
            print(f'No existing vector index found. Loading documents from {document_dir}.')
            for doc in document_path:
                for line in open(os.path.join(document_dir, doc), 'r', encoding='utf-8'):
                    line = line.strip()
                    self.documents.append(line)

            self.embed_model = embed_model
            self.vectors = self.embed_model.get_embeddings(self.documents)

            # 保存向量库
            with open(self.index_file, 'wb') as f:
                pickle.dump((self.vectors, self.documents), f)

            print(f'Loaded {len(self.documents)} documents and created vector index.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        question_vector = self.embed_model.get_embeddings([question])[0]
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()


class LLM:
    def __init__(self, model_path: str, device: str = 'cuda') -> None:
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        print(f'Loading Yuan2.0 model from {model_path} on {self.device}.')

    def generate(self, question: str, context: List):
        if context:
            prompt = f'''作为专业的反诈骗AI助手，你的角色是保护用户免受电信诈骗的侵害，你会严格按照要求的格式回答。你拥有广泛的知识库，
            包括各种已识别的诈骗案例和话术。你会对比下面提供的参考资料和问题，从逻辑的角度分析两者的相似性。注意，参考资料提供的是诈骗案例或者诈骗话术。
            如果参考资料与问题在逻辑上存在相似性，则可以判断问题是诈骗信息。若不是诈骗信息，则在回答'不是诈骗信息'之后，根据问题发散思考，给出存在相关性的诈骗案例。
            参考资料：{context}；问题：{question}
            请根据以下格式进行回答：
            - 问题是否为诈骗信息：[是/不是]诈骗信息
            - 分析理由：[详细分析]
            - 参考资料使用情况：[是否引用，引用内容摘要]
            - 用户建议：[具体的预防措施或进一步行动建议]
            '''
        else:
            prompt = question
        print(prompt)
        prompt += "<sep>"
        print(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        outputs = self.model.generate(inputs, do_sample=False, max_length=1024)
        output = self.tokenizer.decode(outputs[0])
        output = output.replace('<eod>', '')
        print(output)
        return output.split("<sep>")[-1]

def ragResults(embed_model_path, document_dir, document_path, model_path, data):
    print("> Create embedding model...")
    embed_model = EmbeddingModel(embed_model_path)

    print("> Create index...")
    index = VectorStoreIndex(document_dir, document_path, embed_model, index_file='vector_index.pkl')

    question = data
    context = index.query(question)
    print('> Context:', context)

    print("> Create Yuan2.0 LLM...")
    llm = LLM(model_path)

    print('> With RAG:')
    torch.cuda.empty_cache()
    resp = llm.generate(question, context)

    return resp
