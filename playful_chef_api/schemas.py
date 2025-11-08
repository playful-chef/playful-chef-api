from pydantic import BaseModel
from typing import Optional, List


# Ingredient Schema
class Ingredient(BaseModel):
    id: int
    name: str
    recipe_count: Optional[int] = None

    class Config:
        from_attributes = True


# Recipe Schema
class Recipe(BaseModel):
    id: int
    title: str
    directions: Optional[str] = None
    link: Optional[str] = None
    source: Optional[str] = None
    site: Optional[str] = None
    ingredients: List[Ingredient] = []

    class Config:
        from_attributes = True
