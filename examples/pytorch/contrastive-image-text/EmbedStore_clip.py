'''
A class to store embeddings. It stores faiss index, and a dictionary.

'''
import faiss
from pathlib import Path
import pickle
import torch
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
default_model = "openai/clip-vit-base-patch32"
local_root = "/home/ming/dev/transformers/examples/pytorch/contrastive-image-text/embeds"

class EmbedStore():
    def __init__(self, folder_name, model = default_model):
        self.folder_name = folder_name
        self.model_path = model
        self.faiss_index = None

        self.processor = CLIPProcessor.from_pretrained(default_model)
        if model == default_model:
            self.model = CLIPModel.from_pretrained(default_model)
        else:
            self.model = CLIPModel.from_pretrained(f'{local_root}/{model}')
        self.model.to(device)
        self.load_faiss_index(folder_name)
        pass

    def load_faiss_index(self, folder_name):
        base = str(Path(__file__).parent)
        if self.model_path == default_model:
            index_path = f'{base}/embeds/clip-vit-base-patch32/{folder_name}.pickle'
        else:
            index_path = f'{base}/embeds/{self.model_path}/{folder_name}.pickle'

        # if index_path does not exist, return
        if not os.path.exists(index_path):
            return
        
        with open(index_path, 'rb') as handle:
            embs = pickle.load(handle)

        self.file_path = {}
        self.faiss_index = faiss.IndexFlatL2(embs[list(embs.keys())[0]].shape[1])
        cur = 0
        for k, v in embs.items():
            faiss.normalize_L2(v)
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
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(device)
        outputs = self.model.get_text_features(**inputs, output_attentions=True)
        outputs = outputs.detach().cpu().numpy()

        faiss.normalize_L2(outputs)
        D, I = self.faiss_index.search(outputs, top_k)
        
        file_names = []
        for i in I[0]:
            file_names.append(self.file_path[i])

        return file_names, D[0]
    
    def create_embeds(self):
        '''
        Create embeddings for all images in the folder
        '''
        src_folder = f'/mnt/e/ML.Data/coco_extracted/{self.folder_name}'
        
        embs = {}
        for file in os.listdir(src_folder):
            if file.endswith(".jpg"):
                img = Image.open(f'{src_folder}/{file}')
                inputs = self.processor(images=img, return_tensors="pt", padding=True).to(device)
                outputs = self.model.get_image_features(**inputs, output_attentions=True)
                outputs = outputs.detach().cpu().numpy()
                embs[file] = outputs

                if len(embs) % 1000 == 0:
                    print(f'{self.folder_name}: processed {len(embs)} images', flush=True)

        #save embeds to a pickle file
        with open(f'{local_root}/{self.model_path}/{self.folder_name}.pickle', 'wb') as handle:
            pickle.dump(embs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
        