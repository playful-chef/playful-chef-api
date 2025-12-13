"""
Добавление проверки релевантности к уже существующим результатам оценки
Читает evaluation_results.json и добавляет метрики релевантности
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
import openai

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

client = openai.OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=os.getenv("LLM_API_KEY")
)


def check_ingredient_relevance(ingredients: list[str], recipe_title: str) -> bool:
    """Проверяет, содержит ли рецепт ВСЕ указанные ингредиенты"""
    prompt = f"""Дан список ингредиентов: {', '.join(ingredients)}.
Найденный рецепт: "{recipe_title}".

Вопрос: Может ли этот рецепт содержать ВСЕ указанные ингредиенты?
Ответь ТОЛЬКО "Yes" или "No" без объяснений."""

    try:
        response = client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"Ошибка проверки: {e}")
        return False


def check_name_relevance(query: str, recipe_title: str) -> bool:
    """Проверяет семантическое сходство запроса и названия рецепта"""
    prompt = f"""Запрос пользователя: "{query}".
Найденное блюдо: "{recipe_title}".

Вопрос: Соответствует ли найденное блюдо запросу пользователя?
Ответь ТОЛЬКО "Yes" или "No" без объяснений."""

    try:
        response = client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer
    except Exception as e:
        print(f"Ошибка проверки: {e}")
        return False


def extract_ingredients_from_query(query: str) -> list[str]:
    """Извлекает список ингредиентов из запроса"""
    if ',' in query:
        return [ing.strip() for ing in query.split(',')]

    if query.startswith("что"):
        prompt = f"""Запрос: "{query}"
Извлеки ВСЕ ингредиенты из этого запроса.
Верни только список ингредиентов через запятую, без других слов."""

        try:
            response = client.chat.completions.create(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            ingredients_str = response.choices[0].message.content.strip()
            return [ing.strip() for ing in ingredients_str.split(',')]
        except:
            return []

    return []


def extract_recipe_title(response_text: str) -> str:
    """Извлекает название первого рецепта из ответа"""
    lines = response_text.strip().split('\n')
    if lines:
        return lines[0].strip()
    return ""


def add_relevance_check():
    """Добавляет проверку релевантности к существующим результатам"""

    print("\n" + "="*70)
    print("ДОБАВЛЕНИЕ ПРОВЕРКИ РЕЛЕВАНТНОСТИ")
    print("="*70 + "\n")

    # Загрузка существующих результатов
    print("Загрузка результатов из evaluation_results.json...")
    with open('test_validate/evaluation_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['detailed_results']
    print(f"Найдено результатов: {len(results)}\n")

    # Метрики релевантности
    db_relevant_count = 0
    rag_relevant_count = 0
    db_total = 0
    rag_total = 0

    print("-"*70)
    print("ПРОВЕРКА РЕЛЕВАНТНОСТИ:\n")

    # Проверка каждого результата
    for i, result in enumerate(results, 1):
        if 'error' in result:
            result['is_relevant'] = None
            continue

        query = result['query']
        expected_tool = result['expected_tool']
        detected_tool = result['detected_tool']
        has_response = result['has_response']
        tool_correct = result['tool_correct']

        print(f"[{i}/{len(results)}] {query}")

        is_relevant = False

        # Проверяем релевантность только если инструмент выбран правильно и есть ответ
        if has_response and tool_correct:
            recipe_title = extract_recipe_title(result['response'])

            if expected_tool == "db":
                ingredients = extract_ingredients_from_query(query)
                if ingredients:
                    is_relevant = check_ingredient_relevance(ingredients, recipe_title)
                    db_total += 1
                    if is_relevant:
                        db_relevant_count += 1
                    print(f"  Ингредиенты: {ingredients}")
                    print(f"  Рецепт: {recipe_title}")
                    print(f"  Релевантность: {'Yes' if is_relevant else 'No'}")

            elif expected_tool == "rag":
                is_relevant = check_name_relevance(query, recipe_title)
                rag_total += 1
                if is_relevant:
                    rag_relevant_count += 1
                print(f"  Рецепт: {recipe_title}")
                print(f"  Релевантность: {'Yes' if is_relevant else 'No'}")
        else:
            print(f"  Пропущен (инструмент: {detected_tool}, ответ: {has_response})")

        result['is_relevant'] = is_relevant
        print()

    # Итоговые метрики
    print("="*70)
    print("ИТОГОВЫЕ МЕТРИКИ РЕЛЕВАНТНОСТИ")
    print("="*70 + "\n")

    if db_total > 0:
        db_relevance = (db_relevant_count / db_total) * 100
        print(f"SQL поиск (по ингредиентам):")
        print(f"  Релевантность: {db_relevant_count}/{db_total} ({db_relevance:.1f}%)")
        print(f"  Процент рецептов, содержащих ВСЕ указанные ингредиенты\n")
    else:
        db_relevance = 0
        print("SQL поиск: нет результатов для проверки\n")

    if rag_total > 0:
        rag_relevance = (rag_relevant_count / rag_total) * 100
        print(f"RAG поиск (по названию):")
        print(f"  Релевантность: {rag_relevant_count}/{rag_total} ({rag_relevance:.1f}%)")
        print(f"  Процент рецептов, название которых соответствует запросу\n")
    else:
        rag_relevance = 0
        print("RAG поиск: нет результатов для проверки\n")

    # Обновление метрик
    data['metrics']['db_relevance'] = db_relevance
    data['metrics']['rag_relevance'] = rag_relevance
    data['detailed_results'] = results

    # Сохранение обновленных результатов
    with open('test_validate/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Результаты обновлены в evaluation_results.json\n")


if __name__ == "__main__":
    add_relevance_check()
