'''
A class to store embeddings. It stores faiss index, and a dictionary.

'''
import faiss
from pathlib import Path
import pickle
import torch
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class EmbedStore():
    def __init__(self, folder_name):
        self.load_faiss_index(folder_name)
        pass

    def load_faiss_index(self, folder_name):
        base = str(Path(__file__).parent)
        with open(f'{base}/embeds/{folder_name}.pickle', 'rb') as handle:
            embs = pickle.load(handle)

        self.file_path = {}
        self.faiss_index = faiss.IndexFlatL2(embs[list(embs.keys())[0]].shape[1])
        cur = 0
        for k, v in embs.items():
            self.faiss_index.add(v)
            self.file_path[cur] = k
            cur += 1

        assert self.faiss_index.is_trained
        return
        

    def search_faiss_index(self, text, top_k=10):
        '''
        The store support retrieval of embeddings by text: 
        - create embeding from the text
        - search the faiss index for the nearest neighbor
        - use the dictionary to locate the image file name
        - return the image file name and the distance
        '''
        inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
        outputs = model.get_text_features(**inputs, output_attentions=True)
        outputs = outputs.detach().cpu().numpy()

        D, I = self.faiss_index.search(outputs, top_k)
        
        file_names = []
        for i in I[0]:
            file_names.append(self.file_path[i])

        return file_names, D[0]
        