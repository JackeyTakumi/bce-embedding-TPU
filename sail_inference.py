import sophon.sail as sail
from transformers import AutoTokenizer
import numpy as np

bmodel_path = './models/bm1684x/bce-embedding-base_v1.bmodel'
dev_id = 0

engine = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
graph_name = engine.get_graph_names()[0]
input_names = engine.get_input_names(graph_name)
output_names = engine.get_output_names(graph_name)

sentences = ['如何更换花呗绑定银行卡', '算能2023年第四季度销售会议通报', '两化企业管理方法TKP', '比特家校招文化分享会通报']
print('input sentences: ', sentences)
tokenizer = AutoTokenizer.from_pretrained('./token_config')

encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")

input_ids = encoded_input['input_ids']
print('input tokens: ', input_ids)
attention_mask = encoded_input['attention_mask']
input_ids, attention_mask = input_ids.numpy(), attention_mask.numpy()

if input_ids.shape[1] > 512:
    input_ids = input_ids[:, :512]
    attention_mask = attention_mask[:, :512]
elif input_ids.shape[1] < 512:
    input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=0)
    attention_mask = np.pad(attention_mask,
                            ((0, 0), (0, 512 - attention_mask.shape[1])),
                            mode='constant', constant_values=0)
    
input_data = { input_names[0]: input_ids, input_names[1]: attention_mask }
outputs = engine.process(graph_name, input_data)

embeddings = outputs[output_names[0]][:, 0]
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print('first 10 digits of every sentence embedding: ', [x[:10] for x in embeddings])

