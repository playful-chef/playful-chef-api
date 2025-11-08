from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from playful_chef_api import models
from playful_chef_api.models import Recipe, recipe_ingredient, Ingredient


def get_random_recipes(db: Session, limit: int = 10):
    """Get random recipes from the database with their ingredients"""
    return (
        db.query(models.Recipe)
        .order_by(func.random())
        .limit(limit)
        .options(joinedload(models.Recipe.ingredients))
        .all()
    )


def get_recipe_by_id(db: Session, id: int):
    """Get random recipes from the database with their ingredients"""
    return (
        db.query(models.Recipe)
        .where(Recipe.id == id)
        .options(joinedload(models.Recipe.ingredients))
        .first()
    )


def get_recipes_by_ingredients(
    db: Session, ingredient_names: list[str], cutoff: float = 0.5, limit: int = 10
):
    """
    Returns recipes where at least 'cutoff' ingredients are in ingredient_names.

    Args:
        db: Database session
        ingredient_names: List of ingredient names to match
        cutoff: Minimum percentage of matching ingredients (0.0 to 1.0)
        limit: Maximum number of recipes to return

    Returns:
        List of Recipe objects that meet the cutoff criteria
    """

    # Subquery to count total ingredients per recipe
    total_ingredients_subq = (
        db.query(
            Recipe.id.label("recipe_id"),
            func.count(recipe_ingredient.c.ingredient_id).label("total_ingredients"),
        )
        .join(recipe_ingredient, Recipe.id == recipe_ingredient.c.recipe_id)
        .group_by(Recipe.id)
        .subquery()
    )

    # Subquery to count matching ingredients per recipe
    matching_ingredients_subq = (
        db.query(
            Recipe.id.label("recipe_id"),
            func.count(recipe_ingredient.c.ingredient_id).label("matching_ingredients"),
        )
        .join(recipe_ingredient, Recipe.id == recipe_ingredient.c.recipe_id)
        .join(Ingredient, recipe_ingredient.c.ingredient_id == Ingredient.id)
        .filter(Ingredient.name.in_(ingredient_names))
        .group_by(Recipe.id)
        .subquery()
    )

    # Main query to find recipes that meet the cutoff
    recipes = (
        db.query(Recipe)
        .join(total_ingredients_subq, Recipe.id == total_ingredients_subq.c.recipe_id)
        .outerjoin(
            matching_ingredients_subq,
            Recipe.id == matching_ingredients_subq.c.recipe_id,
        )
        .filter(
            # Handle case where no ingredients match (matching_ingredients is NULL)
            func.coalesce(matching_ingredients_subq.c.matching_ingredients, 0)
            / total_ingredients_subq.c.total_ingredients
            >= cutoff
        )
        .order_by(
            # Order by match percentage descending, then by recipe id
            (
                func.coalesce(matching_ingredients_subq.c.matching_ingredients, 0)
                / total_ingredients_subq.c.total_ingredients
            ).desc(),
            Recipe.id,
        )
        .limit(limit)
        .all()
    )

    return recipes


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
