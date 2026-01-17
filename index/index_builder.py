import pandas as pd
import numpy as np
from typing import List
from langchain_community.vectorstores import FAISS
from light_embed import TextEmbedding
from tqdm import tqdm
from langchain_core.documents import Document

# Колонки, которые убираем перед индексированием (как в подготовке БД)
COLUMNS_TO_DROP = [
    "captured_at",
    "author",
    "equipment",
    "protein_grams",
    "fat_grams",
    "carb_grams",
    "calories",
    "calories_total",
]


class LightEmbedWrapper:
    def __init__(self, model: TextEmbedding):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Assuming TextEmbedding.embed returns a list of embeddings for a list of texts
        embeddings = []
        for text in tqdm(texts, desc="Embedding documents"):
            # Assuming TextEmbedding.embed can handle a single text
            embedding = self.model.encode([text])[0]
            embeddings.append(embedding)
        return embeddings


def create_text_for_embedding(row: dict) -> str:
    """Собрать одну строку с данными рецепта для эмбеддинга."""
    text_parts = [
        f"Title: {row.get('title', '')}",
        f"Description: {row.get('description', '')}",
        f"Categories: {row.get('categories', '')}",
        f"Ingredients: {row.get('ingredients', '')}",
        f"Directions: {row.get('directions', '')}",
        f"Total time: {row.get('total_time', '')}",
        f"Servings: {row.get('servings', '')}",
        (
            "Macros per 100g: "
            f"protein {row.get('protein_percent', '')}, "
            f"fat {row.get('fat_percent', '')}, "
            f"carb {row.get('carb_percent', '')}, "
            f"calories {row.get('calories_per_100g', '')}"
        ),
    ]
    return " ".join(text_parts)


class RecipeVectorDB:
    def __init__(self, model_path: str):
        self.model = LightEmbedWrapper(
            TextEmbedding(
                model_path,
                model_config={"onnx_file": "model.onnx"},
                cache_folder="index/sentence-transformers",
            )
        )
        self.index = None
        self.recipes_data = None

    def create_index(self, df: pd.DataFrame):
        """Создать FAISS-индекс из датафрейма."""

        # Выравниваем с обработкой БД, чтобы id совпадали с SQLite
        df = df.rename(columns={"instructions": "directions", "url": "link"})
        df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
        df = df.dropna()
        df["id"] = df.index.copy()

        # Текст для эмбеддинга
        df["embedding_text"] = df.apply(create_text_for_embedding, axis=1)

        # Храним записи для search_similar (резерв)
        self.recipes_data = df.to_dict("records")

        documents = []
        for row in df.itertuples(index=False):
            metadata = {
                "id": row.id,
                "title": row.title,
                "url": row.link,
                "description": row.description,
                "ingredients": row.ingredients,
                "categories": row.categories,
                "directions": row.directions,
                "total_time": row.total_time,
                "servings": row.servings,
            }
            documents.append(
                Document(page_content=row.embedding_text, metadata=metadata)
            )

        # Собираем и сохраняем FAISS-индекс
        print("Создаем FAISS индекс")
        self.index = FAISS.from_documents(documents, self.model)
        self.index.save_local("index/faiss_index")
        print("Индекс сохранен")

    def search_similar(self, query: str, k: int = 5):
        """Return k most similar recipes."""
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        distances, indices = self.index.search(query_embedding.astype("float32"), k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.recipes_data):
                recipe = self.recipes_data[idx].copy()
                recipe["similarity_score"] = float(dist)
                results.append(recipe)

        return results

    def load_db(self, index_path: str):
        return FAISS.load_local(
            index_path, self.model, allow_dangerous_deserialization=True
        )  # allow_dangerous_deserialization нужен, чтобы подхватить метаданные FAISS


if __name__ == "__main__":
    print("Запускаем сборку индекса")
    df = pd.read_csv("./data/recipe-parser/data/output/recipes.tsv", sep="\t")
    index_builder = RecipeVectorDB(
        model_path="onnx-models/paraphrase-multilingual-MiniLM-L12-v2-onnx"
    )
    index_builder.create_index(df=df)
