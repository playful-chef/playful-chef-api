from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from typing import List
from langgraph.prebuilt import create_react_agent
import yaml
import openai
import os
from light_embed import TextEmbedding
from playful_chef_api import crud, schemas


with open("playful_chef_api/config.yml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

url = config["url"]
api_key = os.getenv("LLM_API_KEY")
llm_model = config["llm_model"]
index_path = config["index_path"]
embedder_path = config["embedder_path"]


class RagInput(BaseModel):
    query: str = Field(
        description="""Поисковый запрос для векторной базы (RAG).

        ВАЖНО: Учитывай контекст запроса!

        Примеры ПРАВИЛЬНОГО преобразования:
        • "паста" → "итальянская паста спагетти макароны"
        • "рецепт салата" → "салат овощной легкий"
        • "низкокалорийные блюда" → "низкокалорийные диетические блюда"
        • "суп" → "суп горячий первый блюдо"

        Примеры НЕПРАВИЛЬНОГО:
        • "паста" → "паста" (слишком коротко, будет "паста из...")
        • "салат" → "салат" (будет "салат из...")

        Правила:
        1. Если запрос ОДНО слово → добавь 2-3 синонима/контекста
        2. Если запрос ОБЩИЙ → уточни его
        3. Сохраняй оригинальный смысл
        4. Избегай слишком коротких запросов
        """,
        examples=[
            "паста карбонара спагетти итальянская",
            "салат овощной легкий свежий",
            "низкокалорийные диетические блюда",
            "суп куриный горячий бульон",
        ],
    )


class DataBaseInput(BaseModel):
    raw_query: str = Field(description="Исходный запрос пользователя")
    ingredient_names: list[str] = Field(
        description="""Список ингредиентов для поиска в традиционной БД.

        ---КРИТИЧЕСКИЕ ПРАВИЛА---
        - Добавляй к ингредиентам синонимы
        - Добавляй ингредиенты, которые есть у всех (соль, сахар, масло)

        Примеры ПРАВИЛЬНОГО преобразования:
        • "Что приготовить из курицы и риса?"
            → ["курица", "куриные грудки", "куриное филе", "рис"]
        • "Рецепты с яйцами" → ["яйца", "яйцо", "куриные яйца"]
        • "картофель, лук, морковь" → ["картофель", "лук", "морковь"]
        • "Блюда с говядиной" → ["говядина"]
        • "паста с томатным соусом" → ["паста", "макароны", "томат", "помидор"]
        """,
        examples=[
            ["курица", "рис"],
            ["яйца", "мука", "молоко"],
            ["картофель", "лук", "морковь", "мясо"],
            ["паста", "сыр", "томат", "помидор"],
        ],
    )


class RagResponseFormat(BaseModel):
    dish_id: int = Field(..., description="Номер самого подходящего блюда")


class RAGAgent:
    def __init__(self, index_path, embedder_path):
        print("Инициализируем эмбеддер...")
        model_name = "onnx-models/paraphrase-multilingual-MiniLM-L12-v2-onnx"
        self.embedder = TextEmbedding(
            model_name,
            model_config={"onnx_file": "model.onnx"},
            cache_folder="index/sentence-transformers",
        )
        print("Эмбеддер готов")

        self.index = FAISS.load_local(
            index_path,
            lambda a: self.embedder.encode(a),
            allow_dangerous_deserialization=True,
        )  # загрузка локальной бд

    def go_rag(self, query: str, k=3):
        if isinstance(query, list):
            query = " ".join(query)
        docs = self.index.as_retriever().invoke(query, k=k)
        return docs


class RecipeAgent:
    def __init__(self):
        self.db = None
        self.llm = ChatOpenAI(
            base_url=url, api_key=api_key, model=llm_model, temperature=0.5
        )
        self.client = openai.OpenAI(base_url=url, api_key=api_key)

        # Инициализация RAG
        self.rag_agent = RAGAgent(index_path=index_path, embedder_path=embedder_path)

        # Создаем инструменты
        self.rag_tool = self._create_rag_tool()
        self.tools = [self.rag_tool, self._create_db_tool()]

        # Создаем агента
        self.agent = create_react_agent(
            model=self.llm, tools=self.tools, prompt=config["agent_prompt"]
        )

    def pick_one_dish_id(self, query: str, dishes: List) -> int:
        print("pick_best_dish", query, len(dishes))
        system_prompt = config["choose_one_recipe_prompt"]
        params = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Вопрос пользователя {query}\nСписок блюд: {dishes}",
                },
            ],
            "temperature": 0.3,
            "max_tokens": 200,
        }
        response = self.client.chat.completions.parse(
            **params, response_format=RagResponseFormat
        )
        return response.choices[0].message.parsed.dish_id

    def _create_rag_tool(self):
        """Создает инструмент для RAG поиска"""

        @tool(
            "get_recipes_from_rag",
            args_schema=RagInput,
            return_direct=True,
            parse_docstring=True,
            description=config["get_recipes_from_rag_description"],
        )
        def get_recipes_from_rag(query: str) -> dict | None:
            print("get_recipes_from_rag", query)

            context = self.rag_agent.go_rag(query=query)
            dishes = [i.page_content for i in context]

            if dishes == []:
                return """
                    No recipes found in DB.
                    Try searching by ingredients using get_recipes_from_rag.
                """

            dish_id = self.pick_one_dish_id(query, dishes)
            best_dish = context[dish_id]

            return self._build_recipe_payload(best_dish)

        return get_recipes_from_rag

    def _create_db_tool(self):
        """Создает инструмент для поиска по БД"""

        @tool(
            "get_recipes_from_db",
            args_schema=DataBaseInput,
            return_direct=True,
            parse_docstring=True,
            description=config["get_recipes_from_db_description"],
        )
        def get_recipes_from_db(
            ingredient_names: List[str], raw_query: str
        ) -> schemas.Recipe | None:
            print("get_recipes_from_db", raw_query, ingredient_names)

            # Вызываем функцию поиска
            response = crud.get_recipes_by_ingredients(
                self.db, ingredient_names=ingredient_names
            )
            parsed_recipes = [schemas.Recipe.model_validate(r) for r in response]

            if parsed_recipes == []:
                return self.rag_tool.invoke(raw_query)
            top_recipe_id = self.pick_one_dish_id(raw_query, parsed_recipes)

            return parsed_recipes[top_recipe_id].model_dump()

        return get_recipes_from_db

    def invoke(self, inputs: dict, db):
        """Вызов агента"""
        self.db = db
        return self.agent.invoke(inputs)

    def _build_recipe_payload(self, doc) -> dict:
        """Вернуть структурированный рецепт: из БД по id или из индекса."""
        metadata = getattr(doc, "metadata", {}) or {}
        recipe_id = metadata.get("id")

        recipe = None
        if recipe_id is not None and self.db is not None:
            try:
                recipe = crud.get_recipe_by_id(self.db, id=int(recipe_id))
            except Exception:
                recipe = None

        if recipe:
            return schemas.Recipe.model_validate(recipe).model_dump()

        # Резерв: берем данные из метаданных индекса
        return {
            "id": recipe_id,
            "title": metadata.get("title"),
            "link": metadata.get("url"),
            "directions": metadata.get("directions"),
            "ingredients": metadata.get("ingredients"),
            "description": metadata.get("description"),
            "categories": metadata.get("categories"),
            "total_time": metadata.get("total_time"),
            "servings": metadata.get("servings"),
            "source_url": metadata.get("url"),
        }
