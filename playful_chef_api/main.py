import os
from typing import List, Optional

from fastapi import FastAPI, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from playful_chef_api import models, schemas, crud
from playful_chef_api.database import engine, get_db
from playful_chef_api.model import RecipeAgent

# Отложенная инициализация агента, чтобы сервер поднимался без LLM_API_KEY
Agent: RecipeAgent | None = None

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI application instance
app = FastAPI(
    title="Recipe API",
    description="A FastAPI application for recipes with SQLite database",
    version="1.0.0",
)
inputs = {"messages": []}


@app.get("/agent", response_model=schemas.AgentMessage)
async def get_agent_recipes(
    db: Session = Depends(get_db),
    user_message: str = Query(..., description="РЎР?Р?Р+С%РчР?РёРч РїР?Р>С?Р·Р?Р?Р°С'РчР>С?"),
    user_id: int = Query(..., description="ID РїР?Р>С?Р·Р?Р?Р°С'РчР>С?"),
):
    global Agent
    if Agent is None:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="LLM_API_KEY не задан: эндпоинт /agent недоступен локально.",
            )
        Agent = RecipeAgent()

    inputs = {"messages": [{"role": "user", "content": user_message}]}

    response = Agent.invoke(inputs, db)

    return schemas.AgentMessage(
        user_message=user_message,
        user_id=user_id,
        agent_response=response["messages"][-1].content,
    )


@app.get("/recipes", response_model=List[schemas.Recipe])
async def get_random_recipes(
    db: Session = Depends(get_db),
    limit: int = 10,
    ingredients: Optional[List[str]] = Query(
        None, description="Filter recipes by ingredients"
    ),
):
    """
    Get random recipes from the database.

    - **limit**: Number of random recipes to return (default: 10)
    - **ingredients**: Optional list of ingredients to filter recipes
    """
    if ingredients:
        print(ingredients)
        recipes = crud.get_recipes_by_ingredients(
            db, ingredient_names=ingredients, limit=limit
        )
    else:
        recipes = crud.get_random_recipes(db, limit=limit)

    return recipes


@app.get("/recipes/{id}", response_model=schemas.Recipe)
async def get_recipe_by_id(id: int, db: Session = Depends(get_db)):
    return crud.get_recipe_by_id(db, id=id)


@app.get("/ingredients", response_model=List[schemas.Ingredient])
async def get_ingredients(db: Session = Depends(get_db)):
    """
    Get random recipes from the database.

    - **limit**: Number of random recipes to return (default: 10)
    """
    ingredients = crud.get_all_ingredients(db)
    return ingredients
