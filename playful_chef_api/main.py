from fastapi import FastAPI, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from playful_chef_api import models, schemas, crud
from playful_chef_api.database import engine, get_db
from playful_chef_api.model import RAGAgent


Agent = RAGAgent()

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI application instance
app = FastAPI(
    title="Recipe API",
    description="A FastAPI application for recipes with SQLite database",
    version="1.0.0",
)
inputs = {"messages": []}


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

    Agent.go_rag(ingredients)

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
