# 导入所需的库
import os.path
from typing import List
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

# 定义向量模型类
class EmbeddingModel:
    """
    class for EmbeddingModel
    """

    def __init__(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model = AutoModel.from_pretrained(path).cuda()
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List) -> List[float]:
        """
        calculate embedding for text list
        """
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()

# 定义向量库索引类
class VectorStoreIndex:
    """
    class for VectorStoreIndex
    """

    def __init__(self, document_dir: str, document_path: List, embed_model: EmbeddingModel) -> None:
        self.documents = []
        for doc in document_path:
            #print(os.path.join(document_dir, doc))
            for line in open(os.path.join(document_dir, doc), 'r', encoding='utf-8'):
                line = line.strip()
                self.documents.append(line)

        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)

        print(f'Loading {len(self.documents)} documents for {document_path}.')

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self, question: str, k: int = 1) -> List[str]:
        question_vector = self.embed_model.get_embeddings([question])[0]
        result = np.array([self.get_similarity(question_vector, vector) for vector in self.vectors])
        return np.array(self.documents)[result.argsort()[-k:][::-1]].tolist()

# 定义大语言模型类
class LLM:
    """
    class for Yuan2.0 LLM
    """

    def __init__(self, model_path: str) -> None:
        print("Creat tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

        print("Creat model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

        print(f'Loading Yuan2.0 model from {model_path}.')

    def generate(self, question: str, context: List):
        if context:
            prompt = f'你是一名经验丰富的警察，你对电信诈骗的案件了如指掌，对每一个向你咨询的人都十分热情 且乐于给出充分且详细的建议。请你基于给出的参考资料来回答输入中的问题或者是否为诈骗信息。背景：{context}\n问题：{question}\n请基于背景，回答问题。'
        else:
            prompt = question

        prompt += "<sep>"
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        outputs = self.model.generate(inputs, do_sample=False, max_length=1024)
        output = self.tokenizer.decode(outputs[0])
        self.model.cpu()
        torch.cuda.empty_cache()
        #print(output.split("<sep>")[-1])
        return output.split("<sep>")[-1]

def ragResults(embed_model_path, document_dir, document_path, model_path, data):
    print("> Create embedding model...")
    embed_model = EmbeddingModel(embed_model_path)

    print("> Create index...")
    index = VectorStoreIndex(document_dir, document_path, embed_model)

    question = data
    context = index.query(question)
    print('> Context:', context)

    print("> Create Yuan2.0 LLM...")
    llm = LLM(model_path)

    print('> With RAG:')
    torch.cuda.empty_cache()
    resp = llm.generate(question, context)

    return resp