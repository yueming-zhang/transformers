import faiss
import pickle
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from EmbedStore import EmbedStore

def test_create_embeds():
    store = EmbedStore('train2017', 'checkpoint-13500')
    store.create_embeds()
    pass

def test_search_embeds():
    store = EmbedStore('val2017', 'checkpoint-13500')
    res = store.search_faiss_index('a dog')
    assert res  is not None
    pass