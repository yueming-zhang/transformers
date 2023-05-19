import faiss
import pickle
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from inference import image_classification, get_txt_embeds, get_image_embeds_folder, load_faiss_index, search_faiss_index
from EmbedStore import EmbedStore

def test_clip_image_classification():

    ret = image_classification()
    assert ret[0] > ret[1]

def test_image_embeds():

    folder_name = 'train2017'
    folder = f'/mnt/e/ML.Data/coco_extracted/{folder_name}'
    embs = get_image_embeds_folder(folder)

    #save embeds to a pickle file
    with open(f'{folder_name}.pickle', 'wb') as handle:
        pickle.dump(embs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def test_load_embeds():
    store = EmbedStore('train2017')

    ret = store.search_faiss_index(['two dogs playing in the snow'])
    pass