import json
import time
import os
from pathlib import Path
from sqlalchemy.orm import Session
from playful_chef_api.database import SessionLocal
from playful_chef_api.model import RecipeAgent


def detect_tool_used(agent_response: str) -> str:
    """
    Определяет какой инструмент был использован по логам
    В реальности нужно логировать это в agent.invoke()

    Пока используем эвристику:
    - Если в ответе несколько рецептов (список) -> скорее всего DB
    - Если один рецепт с описанием -> скорее всего RAG
    """
    # Простая эвристика: если в ответе есть несколько строк с \n\n -> это список из DB
    if agent_response.count('\n\n') >= 2 or agent_response.count('http') > 2:
        return "db"
    else:
        return "rag"


def run_evaluation():
    """Запуск оценки системы"""

    # Загрузка тестовых данных
    print(f"\n{'='*70}")
    print(f"ЗАПУСК ОЦЕНКИ СИСТЕМЫ ПОИСКА РЕЦЕПТОВ")
    print(f"{'='*70}\n")

    with open('test_validate/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    queries = test_data['test_queries']
    print(f"Загружено тестовых запросов: {len(queries)}\n")

    # Инициализация агента и БД
    print("Инициализация агента...")
    agent = RecipeAgent()
    db = SessionLocal()
    print(f"Агент готов\n")

    # Метрики
    results = []
    tool_selection_correct = 0
    successful_responses = 0
    total_time = 0

    print(f"{'─'*70}")
    print(f"ТЕСТИРОВАНИЕ:\n")

    # Прогон тестов
    for i, test in enumerate(queries, 1):
        query = test['query']
        expected_tool = test['expected_tool']

        print(f"[{i}/{len(queries)}] {query}")
        print(f"    Ожидаемый инструмент: {expected_tool.upper()}")

        try:
            start_time = time.time()
            inputs = {"messages": [{"role": "user", "content": query}]}
            response = agent.invoke(inputs, db)
            elapsed_time = time.time() - start_time

            agent_response = response["messages"][-1].content
            total_time += elapsed_time

            detected_tool = detect_tool_used(agent_response)

            tool_correct = (detected_tool == expected_tool or expected_tool == "any")
            has_response = len(agent_response) > 10  # Минимальная длина ответа

            if tool_correct:
                tool_selection_correct += 1

            if has_response:
                successful_responses += 1

            # Вывод результата
            tool_status = f"Yes" if tool_correct else f"No"
            response_status = f"Yes" if has_response else f"No"

            print(f"    Использованный инструмент: {detected_tool.upper()} {tool_status}")
            print(f"    Ответ получен: {response_status}")
            print(f"    Время ответа: {elapsed_time:.2f}s")
            print(f"    Ответ (первые 100 символов): {agent_response[:100]}...")

            results.append({
                "query": query,
                "expected_tool": expected_tool,
                "detected_tool": detected_tool,
                "tool_correct": tool_correct,
                "has_response": has_response,
                "response_time": elapsed_time,
                "response": agent_response
            })

        except Exception as e:
            print(f"ОШИБКА: {str(e)}")
            results.append({
                "query": query,
                "expected_tool": expected_tool,
                "error": str(e)
            })

        print()

    db.close()

    # Вывод итоговых метрик
    print(f"{'='*70}")
    print(f"ИТОГОВЫЕ МЕТРИКИ")
    print(f"{'='*70}\n")

    total_queries = len(queries)
    tool_accuracy = (tool_selection_correct / total_queries) * 100
    success_rate = (successful_responses / total_queries) * 100
    avg_time = total_time / total_queries

    print(f"1. Tool Selection Accuracy: {tool_selection_correct}/{total_queries} ", end="")
    print(f"({tool_accuracy:.1f}%)")
    print(f"   → Измеряет, насколько точно агент выбирает правильный инструмент (SQL vs RAG)\n")

    print(f"2. Response Success Rate: {successful_responses}/{total_queries} ", end="")
    print(f"({success_rate:.1f}%)")
    print(f"   → Процент запросов, на которые система дала корректный ответ\n")

    print(f"3. Average Response Time: {avg_time:.2f}s")
    print(f"   → Среднее время ответа системы на запрос пользователя\n")

    # Анализ по типам инструментов
    print(f"{'─'*70}")
    print(f"АНАЛИЗ ПО ТИПАМ ИНСТРУМЕНТОВ\n")
    print(f"{'─'*70}")

    db_queries = [r for r in results if r.get('expected_tool') == 'db' and 'error' not in r]
    rag_queries = [r for r in results if r.get('expected_tool') == 'rag' and 'error' not in r]

    if db_queries:
        db_correct = sum(1 for r in db_queries if r.get('tool_correct', False))
        db_accuracy = (db_correct / len(db_queries)) * 100
        print(f"SQL поиск (по ингредиентам):")
        print(f"  Точность: {db_correct}/{len(db_queries)} ({db_accuracy:.1f}%)")
        print(f"  Среднее время: {sum(r.get('response_time', 0) for r in db_queries) / len(db_queries):.2f}s\n")

    if rag_queries:
        rag_correct = sum(1 for r in rag_queries if r.get('tool_correct', False))
        rag_accuracy = (rag_correct / len(rag_queries)) * 100
        print(f"RAG поиск (по описанию/названию):")
        print(f"  Точность: {rag_correct}/{len(rag_queries)} ({rag_accuracy:.1f}%)")
        print(f"  Среднее время: {sum(r.get('response_time', 0) for r in rag_queries) / len(rag_queries):.2f}s\n")

    # Рекомендации
    print(f"{'='*70}")
    print(f"РЕКОМЕНДАЦИИ")
    print(f"{'='*70}\n")

    if tool_accuracy < 70:
        print(f"Tool Selection Accuracy низкая (<70%)")

    if avg_time > 5:
        print(f"Среднее время ответа >5s")

    if success_rate < 90:
        print(f"Success Rate низкий (<90%)")

    if tool_accuracy >= 70 and success_rate >= 90 and avg_time <= 5:
        print(f"Все метрики в норме, система работает хорошо.\n")

    # Сохранение результатов
    with open('test_validate/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            "metrics": {
                "tool_selection_accuracy": tool_accuracy,
                "response_success_rate": success_rate,
                "average_response_time": avg_time,
                "db_accuracy": db_accuracy if db_queries else 0,
                "rag_accuracy": rag_accuracy if rag_queries else 0
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)

    print(f"Детальные результаты сохранены в evaluation_results.json\n")


if __name__ == "__main__":
    print("Работа")
    run_evaluation()
