from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from playful_chef_api import models


def get_random_recipes(db: Session, limit: int = 10):
    """Get random recipes from the database with their ingredients"""
    return (
        db.query(models.Recipe)
        .order_by(func.random())
        .limit(limit)
        .options(joinedload(models.Recipe.ingredients))
        .all()
    )


def get_recipes_by_ingredients(
    db: Session, ingredient_names: list[str], limit: int = 10
):
    """Get recipes that contain any of the specified ingredients"""
    return (
        db.query(models.Recipe)
        .join(models.Recipe.ingredients)
        .filter(models.Ingredient.name.in_(ingredient_names))
        .options(joinedload(models.Recipe.ingredients))
        .order_by(func.random())
        .limit(limit)
        .all()
    )


def get_all_ingredients(db: Session):
    """List ingredients, sorted by usage, excluding single-use ingredients"""
    results = (
        db.query(
            models.Ingredient.id,
            models.Ingredient.name,
            func.count(models.recipe_ingredient.c.recipe_id).label("recipe_count"),
        )
        .join(models.recipe_ingredient)
        .group_by(models.Ingredient.id, models.Ingredient.name)
        .having(func.count(models.recipe_ingredient.c.recipe_id) > 1)
        .order_by(func.count(models.recipe_ingredient.c.recipe_id).desc())
        .all()
    )
    return results
