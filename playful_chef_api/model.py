from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from crud import get_recipes_by_ingredients
from typing import List
from langgraph.prebuilt import create_react_agent
import yaml
import openai


url = "https://api.mistral.ai/v1"
api_key = "vHbKd2pQPuz6H8uCzF7bFv3wYeIB6Yle"
model = "mistral-small-latest"
index_path = "index/faiss_index"
embedder_path = "index/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

llm = ChatOpenAI(base_url=url, api_key=api_key, model=model, temperature=0.5)


with open("playful_chef_api/config.yml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


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
    ingredient_names: list[str] = Field(
        description="""Список ингредиентов для поиска в традиционной БД.

        КРИТИЧЕСКОЕ ПРАВИЛО: Извлекай ТОЛЬКО названия продуктов, никаких предлогов!

        Примеры ПРАВИЛЬНОГО преобразования:
        • "Что приготовить из курицы и риса?" → ["курица", "рис"]
        • "Рецепты с яйцами и молоком" → ["яйца", "молоко"]
        • "Имею картофель, лук, морковь" → ["картофель", "лук", "морковь"]
        • "Блюда с говядиной" → ["говядина"]
        • "паста с томатным соусом" → ["паста", "томатный соус"]
        """,
        examples=[
            ["курица", "рис"],
            ["яйца", "мука", "молоко"],
            ["картофель", "лук", "морковь", "мясо"],
            ["паста", "сыр", "томаты"],
        ],
    )


class RagResponseFormat(BaseModel):
    dish_id: int = Field(..., description="Номер самого подходящего блюда")


class RAGAgent:
    def __init__(self, index_path, embedder_path):
        self.embedder = HuggingFaceEmbeddings(model_name=embedder_path)
        self.index = FAISS.load_local(
            index_path, self.embedder, allow_dangerous_deserialization=True
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
            base_url=url, api_key=api_key, model=model, temperature=0.5
        )
        self.client = openai.OpenAI(base_url=url, api_key=api_key)

        # Инициализация RAG
        self.rag_agent = RAGAgent(index_path=index_path, embedder_path=embedder_path)

        # Создаем инструменты
        self.tools = [self._create_rag_tool(), self._create_db_tool()]

        # Создаем агента
        self.agent = create_react_agent(
            model=llm, tools=self.tools, prompt=config["agent_prompt"]
        )

    def _create_rag_tool(self):
        """Создает инструмент для RAG поиска"""

        @tool(
            "get_recipes_from_rag",
            args_schema=RagInput,
            return_direct=True,
            parse_docstring=True,
            description=config["get_recipes_from_rag_description"],
        )
        def get_recipes_from_rag(query: str) -> str:
            print("get_recipes_from_rag")

            context = self.rag_agent.go_rag(query=query)

            dishes = [i.page_content for i in context]

            system_prompt = """
            У тебя есть список блюд. Выбери самое подходящее под запрос пользователя.
            Верни только номер блюда.
            Номер начинается с 0
            """
            params = {
                "model": model,  # Имя модели для локального сервера
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Вопрос пользователя {query}\n"
                        f"Список блюд: {dishes}",
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 200,
            }
            response = self.client.chat.completions.parse(
                **params, response_format=RagResponseFormat
            )
            dish_id = response.choices[0].message.parsed.dish_id

            best_dish = context[dish_id]

            message = (
                f'{best_dish.metadata["title"]}'
                f'\n{best_dish.metadata["description"]}'
                f'\n{best_dish.metadata["url"]}'
            )

            # system_prompt = config["rag_prompt"]
            # prompt_template = ChatPromptTemplate.from_messages(
            #     [("system", system_prompt), ("human", "{query}")]
            # )
            #
            # formatted_prompt = prompt_template.format_messages(
            #     context=context, query=query
            # )
            #
            # response = self.llm.invoke(formatted_prompt)
            return message

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
        def get_recipes_from_db(ingredient_names: List[str]) -> List:
            print("get_recipes_from_db")

            # Вызываем функцию поиска
            response = get_recipes_by_ingredients(
                self.db, ingredient_names=ingredient_names
            )
            return [f"{i.title}\n{i.link}" for i in response]

        return get_recipes_from_db

    def invoke(self, inputs: dict, db):
        """Вызов агента"""
        self.db = db
        return self.agent.invoke(inputs)
