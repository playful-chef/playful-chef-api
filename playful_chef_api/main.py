from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List

from playful_chef_api import models, schemas, crud
from playful_chef_api.database import engine, get_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Create FastAPI application instance
app = FastAPI(
    title="Recipe API",
    description="A FastAPI application for recipes with SQLite database",
    version="1.0.0",
)


@app.get("/recipes", response_model=List[schemas.Recipe])
async def get_random_recipes(db: Session = Depends(get_db), limit: int = 10):
    """
    Get random recipes from the database.

    - **limit**: Number of random recipes to return (default: 10)
    """
    recipes = crud.get_random_recipes(db, limit=limit)
    return recipes


@app.get("/ingredients", response_model=List[schemas.Ingredient])
async def get_ingredients(db: Session = Depends(get_db)):
    """
    Get random recipes from the database.

    - **limit**: Number of random recipes to return (default: 10)
    """
    ingredients = crud.get_all_ingredients(db)
    return ingredients
