import json
import copy
from gensim.models import KeyedVectors

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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




# 加载预训练的Word2Vec模型
# 替换路径为你下载的模型文件路径
model_path = '/home/user/Documents/ZJ/concept-graphs/edit_graphs/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

file = "/home/user/Documents/ZJ/concept-graphs/Replica/room0/sg_cache/cfslam_object_relations.json"
try:
    with open(file, 'r') as f:
        data_raw = json.load(f)
except FileNotFoundError:
    print("文件未找到，请检查文件路径是否正确。")
except json.JSONDecodeError:
    print("文件内容不是有效的JSON格式。")
object_dict = {}
data = copy.deepcopy(data_raw)
for item in data:
    if item['object1'] ['id'] not in object_dict.keys():
        item['object1']["words_vector"]= sentence_vector(item['object1']["object_tag"],word2vec_model)
        object_dict[item['object1'] ['id']] = item['object1']
    if item['object2'] ['id'] not in object_dict.keys():
        item['object2']["words_vector"]= sentence_vector(item['object2']["object_tag"],word2vec_model)
        object_dict[item['object2'] ['id']] = item['object2']



#读取用户输入
# user_input = input("Please enter your query: ")
text = "I can see a bottle of beverage in the sideboard"

import time
from os import path
import copy
import json
from openai import OpenAI

API_KEY = 'sk-1F3yr67LlKJUTaWnAa7a6063F7094eFc9cFc4b57Ad93073e'
API_HOST = 'https://xiaoai.one/v1'
client = OpenAI(api_key=API_KEY, base_url=API_HOST)

root = "/home/user/Documents/ZJ/concept-graphs/edit_graphs"
sys_f="system_message.txt"
with open(path.join(root,sys_f)) as f:
    sys_=f.readlines()
messages = [ {"role":"system", "content": '\n'.join(sys_)} ]

with open(path.join(root,"relation.txt")) as f:
    msg_user=f.readlines()
messages.append({"role":"user", "content":"An apple in the cabinet"})
messages.append({"role":"assistant", "content":'\n'.join(msg_user)} )

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
import re


messages_temp=copy.deepcopy(messages)

messages_temp.append({"role":"user", "content":text})
answer=ask_gpt_4("",messages_temp)

answer = eval(answer.replace('json','').replace('```',''))

object_1 = answer['object1']
object_2 = answer['object2']
words_vector_1 = sentence_vector(object_1["object_tag"],word2vec_model)
words_vector_2 = sentence_vector(object_2["object_tag"],word2vec_model)
similarity_score = 0.0
flag_0 = 0
id = ""
#将用户对场景的描述中拆解的两个物品分别跟场景中所有物品做相似度计算
for key, value in object_dict.items():
    vector1 = value["words_vector"]
    score_1 = cosine_similarity([vector1], [words_vector_1])[0, 0]
    score_2 = cosine_similarity([vector1], [words_vector_2])[0, 0]
    if score_1>score_2:
        score = score_1
        flag = 1
    else :
        score = score_2
        flag = 2
    if score>similarity_score:
        similarity_score =score
        id = key
        flag_0 = flag
        
print(id)
object_dict[id].pop('words_vector')
if flag_0 ==2:
    answer['object2'] = object_dict[id]
else:
    answer['object1'] = object_dict[id]

data_raw.append(answer)

filename = 'relation_new.json'

# 打开一个文件用于写入
print("Saving query JSON to file...")
with open(filename, "w") as f:
    json.dump(data_raw, f, indent=4)

