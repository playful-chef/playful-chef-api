import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def create_text_for_embedding(row):
    """Создаем текст для векторного представления рецепта"""
    text_parts = [
        f"Название: {row['title']}",
        f"Описание: {row['description']}",
        f"Категория: {row['categories']}",
        f"Ингредиенты: {row['ingredients']}",
        f"Инструкции: {row['instructions']}",
        f"Время приготовления: {row['total_time']} минут",
        f"Порции: {row['servings']}",
        f"Пищевая ценность на 100г: белки {row['protein_grams']}г, "
        f"жиры {row['fat_grams']}г, углеводы "
        f"{row['carb_grams']}г, калории {row['calories_per_100g']}",
    ]
    return " ".join(text_parts)


class RecipeVectorDB:
    def __init__(self, model_path):
        self.model = HuggingFaceEmbeddings(model_name=model_path)
        self.index = None
        self.recipes_data = None

    def create_index(self, df):
        """Создание FAISS индекса из датафрейма"""

        # Сохраняем полные данные рецептов
        self.recipes_data = df.to_dict("records")

        documents = []
        for row in df.itertuples():
            doc = Document(page_content=row.embedding_text)
            doc.metadata["title"] = row.title
            doc.metadata["url"] = row.url
            doc.metadata["description"] = row.description
            doc.metadata["ingredients"] = row.ingredients
            documents.append(doc)

        # Создаем FAISS индекс
        print("Создаем FAISS индекс")
        self.index = FAISS.from_documents(documents, self.model)
        self.index.save_local("faiss_index")
        print("Индекс создан")
        return

    def search_similar(self, query, k=5):
        """Поиск похожих рецептов"""
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

    def load_db(self, index_path):
        db = FAISS.load_local(
            index_path, self.model, allow_dangerous_deserialization=True
        )  # загрузка локальной бд
        return db


print("Загружаем данные")
df = pd.read_csv("../data/recipe-parser/data/output/recipes.tsv", sep="\t")
df["embedding_text"] = df.apply(create_text_for_embedding, axis=1)
index_builder = RecipeVectorDB(
    model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
index_builder.create_index(df=df)
