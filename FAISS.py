import numpy as np
import pandas as pd
import faiss
import os
import openai

class Embedding:
    
    def openai_embedding(self, text_list:list,
                        model:str="text-embedding-ada-002")->list:
        vector = openai.Embedding.create(input = text_list,
                                         model=model)['data'] # type: ignore
        # 官方文档表示：输出维度为1536，最大接收token：8196
        vector_list = []
        for v in vector:
            vector_list.append(v['embedding'])
        return vector_list

    def get_embedding(self, text_array)-> np.ndarray:
        text_list = list(text_array)
        vector_list = self.openai_embedding(text_list)
        return np.array(vector_list, dtype=np.float32)


class FAISS:
    
    def __init__(self, vector_matrix:np.ndarray=np.array([None]), data = pd.DataFrame()):
        self.vector_matrix = vector_matrix
        self.data = data
        self.save_path = "./vector_store"
        self.index = None
        
    def vector_store(self,
                     vector_matrix:np.ndarray=np.array([None]),
                     index_type="IndexFlatL2",
                     save_path:str="./vector_store",
                     save=True):
        # 文本向量库初始添加
        if not vector_matrix:
            vector_matrix = self.vector_matrix 
        vector_matrix = vector_matrix.astype(np.float32)
        dim = vector_matrix.shape[1]
        
        if not os.path.exists('./vector_store'):
            os.mkdir('./vector_store')
            
        if index_type == "IndexFlatL2":
            # 使用欧式距离
            index = faiss.IndexFlatL2(dim)
        else:
            # 使用内积的方式
            index = faiss.IndexFlatIP(dim)
            
        index.add(vector_matrix) # type: ignore
        
        if save:
            self.vector_save(index, save_path)
        self.index = index
        
    def vector_save(self, index,
                    save_path:str="./vector_store"):
        # 文本向量库保存
        faiss.write_index(index,
                          os.path.join(save_path, "index.faiss"))

    def vector_add(self,
                   vector_matirx,
                   index=None,
                   save_path:str="./vector_store",
                   save=True):
        # 文本向量库添加
        vector_matirx = vector_matirx.astype(np.float32)
        if not index:
            index = self.vector_read()
        index.add(vector_matirx) # type: ignore
        if save:
            self.vector_save(index, save_path)
            
        self.index = index


    def vector_read(self, path:str="./vector_store/index.faiss"):
        # 文本向量库读取
        self.index = faiss.read_index(path)
        # return self.index
    
    def vector_search(self,
                      query_vector:np.ndarray,
                      index=None,
                      top_k:int=4,
                      normalize_L2:bool=False):
        # 文本向量搜索
        if not index:
            index = self.index
            
        if normalize_L2:
            faiss.normalize_L2(query_vector)
            
        query_vector = query_vector.astype(np.float32)
        try:
            scores, indices = index.search(query_vector, top_k) # type: ignore
        except Exception as e:
            scores, indices = index.search(query_vector.reshape(1,-1), top_k) # type: ignore
        
        if not self.data.empty:
            
            para_list = self.data.iloc[indices[0], :].values # type: ignore
            
            return para_list
        
        return indices