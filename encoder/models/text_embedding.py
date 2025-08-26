import os, pickle
import scipy.sparse as sp


def load_text_embedding(dataset, llm, train_matrix, is_dense=True):
    root_path = os.getcwd()
    
    # Load the text embedding
    if is_dense:
        user_embedding_path = os.path.join(root_path, f"data/{dataset}/{llm}_users.pkl")
        item_embedding_path = os.path.join(root_path, f"data/{dataset}/{llm}_items.pkl")

        if not os.path.exists(item_embedding_path):
            raise FileNotFoundError(f"Item embedding file not found: {item_embedding_path}")
        
        with open(item_embedding_path, 'rb') as f:
            item_embedding = pickle.load(f).astype('float64')

        if not os.path.exists(user_embedding_path):
            user_embedding = train_matrix.toarray() @ item_embedding / train_matrix.sum(axis=1)
            with open(user_embedding_path, 'wb') as f:
                pickle.dump(user_embedding, f)
        else:
            with open(user_embedding_path, 'rb') as f:
                user_embedding = pickle.load(f)
    else:
        user_embedding = None
        item_embedding_path = os.path.join(root_path, f"data/{dataset}/multihot.pkl")

        with open(item_embedding_path, 'rb') as f:
            item_embedding = pickle.load(f)
    
    return user_embedding, item_embedding
