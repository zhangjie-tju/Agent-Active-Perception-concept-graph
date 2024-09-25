import json
import re
import copy
from gensim.models import KeyedVectors
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from itertools import combinations

model_path = '/home/user/Documents/ZJ/concept-graphs/edit_graphs/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def sentence_vector(sentence, model, num_features=300):
    words = sentence.split()
    feature_vector = np.zeros((num_features,), dtype="float32")
    num_words = 0
    for word in words:
        if word in model:
            num_words += 1
            feature_vector = np.add(feature_vector, model[word])
    if num_words > 0:
        feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

def node_match(n1, n2):
    """比较节点数据，确保标签相对应"""
    vector1 = sentence_vector(n1['label'],word2vec_model)
    vector2 = sentence_vector(n2['label'],word2vec_model)
    sim = cosine_similarity([vector1], [vector2])[0, 0]
    return sim>0.6

def edge_match(e1, e2):
    """比较边数据，确保权重相同"""
    return e1['weight'] == e2['weight']

def most_common_graphs(G,G1):
    """生成并返回图G的所有可能的连通子图"""
    l = min(len(G),len(G1))
    for n in range(l + 1, 1, -1):
        for nodes in combinations(G, n):
            subG = G.subgraph(nodes)
            for nodes1 in combinations(G1, n):
                subG1 = G1.subgraph(nodes1)
                GM = nx.algorithms.isomorphism.DiGraphMatcher(subG1, subG, node_match=node_match, edge_match=edge_match)
                print("两个图同构：" if GM.is_isomorphic() else "两个图不同构")
                if GM.is_isomorphic():
                    return GM,subG,subG1

def build_gpt_message(system = None, examples = None, prompt = None):
    '''system和example为表示文件路径的字符串，prompt为直接输入大模型的prompt
    '''
    message = []
    with open(system, 'r') as file:
        content = file.read()
    message.append({"role":"system", "content": content})
    with open(examples, 'r') as file:
        content_2 = file.read()
    split_content = [item for item in re.split(r'input|output', content_2) if item]
    l = int(len(split_content)/2)
    for i in range(l):
        message.append({"role":"user", "content":split_content[2*i]})
        message.append({"role":"assistant", "content":split_content[2*i+1]})
    message.append({"role":"user", "content":prompt})
    return message


from openai import OpenAI
import time
API_KEY = 'sk-1F3yr67LlKJUTaWnAa7a6063F7094eFc9cFc4b57Ad93073e'
API_HOST = 'https://xiaoai.one/v1'
client = OpenAI(api_key=API_KEY, base_url=API_HOST)


def ask_gpt_4(question=None, messages=None):
    for i in range(35):
        try:
            if messages is None:
                messages = [
                        {"role": "user", "content": question},
                    ]
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None)
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            if i == 34:
                raise ValueError("GPT failed")
            time.sleep(2)
            continue

    raise ValueError("GPT failed")

