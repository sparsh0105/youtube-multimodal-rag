from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# vector db + retriever for the multimodal chunks
def get_vectorstore(video_id, multimodal_chunks):
    """
    Create or load a Chroma vector store for a specific video.
    """

    persist_dir = Path(f"cache/chroma/{video_id}")
    persist_dir.mkdir(parents=True, exist_ok=True)
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load existing vector store if present (check if directory has content)
    if persist_dir.exists() and list(persist_dir.iterdir()):
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding
        )

    # Otherwise, create from multimodal chunks
    texts = [chunk["combined_text"] for chunk in multimodal_chunks]

    metadatas = [
        {
            "start": chunk["start"],
            "end": chunk["end"]
        }
        for chunk in multimodal_chunks
    ]

    ids = [
        f"{chunk['start']}_{chunk['end']}"
        for chunk in multimodal_chunks
    ]

    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=str(persist_dir)
    )

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    # Note: Chroma automatically persists when persist_directory is set
    # No need to call persist() method in newer versions

    return vector_store

def get_retriever(vector_store, k=10):
    return vector_store.as_retriever(search_kwargs={"k": k})
