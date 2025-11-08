from sqlalchemy import Column, Integer, Text, Table, ForeignKey
from sqlalchemy.orm import relationship
from playful_chef_api.database import Base

# Association table for many-to-many relationship between recipes and ingredients
recipe_ingredient = Table(
    "recipe_ingredients",
    Base.metadata,
    Column("recipe_id", Integer, ForeignKey("recipes.id"), primary_key=True),
    Column("ingredient_id", Integer, ForeignKey("ingredients.id"), primary_key=True),
)


class Recipe(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    directions = Column(Text)
    link = Column(Text)
    source = Column(Text)
    site = Column(Text)

    # Relationship to ingredients through the association table
    ingredients = relationship(
        "Ingredient", secondary=recipe_ingredient, back_populates="recipes"
    )


class Ingredient(Base):
    __tablename__ = "ingredients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text, nullable=False, unique=True, index=True)

    # Relationship to recipes through the association table
    recipes = relationship(
        "Recipe", secondary=recipe_ingredient, back_populates="ingredients"
    )
