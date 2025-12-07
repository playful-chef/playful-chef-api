import pandas as pd
import sqlite3
import re

# Read CSV file
df = pd.read_csv("./data/recipe-parser/data/output/recipes.tsv", sep="\t")
print(f"loaded {len(df)} recipes")


# 1. Clean data

# drop irrelevant columns
df = df.drop(columns=["captured_at", "author"])
# drop rare columns
df = df.drop(columns=["equipment"])
# drop duplicate & computable columns
df = df.drop(
    columns=["protein_grams", "fat_grams", "carb_grams", "calories", "calories_total"]
)
# rename columns for compatibility
df = df.rename(columns={"instructions": "directions", "url": "link"})
# drop recipes with missing data
recipes_df = df.dropna().copy()
# add index
recipes_df["id"] = recipes_df.index.copy()

print(f"after cleaninig: {len(recipes_df)} recipes")


# 2. Parse ingredients

# Use a set to collect data, then create DataFrame at the end
recipe_to_ingredient = list()

for id, row in recipes_df.iterrows():
    ingredient_items = row["ingredients"].split(",")
    for ingredient in ingredient_items:
        (name, *qty_maybe) = ingredient.strip().split(" - ")
        name = name.lower()
        qty_str = "по вкусу" if qty_maybe == [] else qty_maybe[0]
        pattern = r"(?P<value>\d+\.?\d*)\s*(?P<unit>.+)"
        match = re.search(pattern, qty_str)
        qty_num = float(match.group("value")) if match else 0
        unit = match.group("unit") if match else None
        recipe_to_ingredient.append(
            {"ingredient": name, "recipe_id": id, "qty": qty_num, "unit": unit}
        )

recipe_to_ingredient = pd.DataFrame(recipe_to_ingredient)
ingredients = recipe_to_ingredient[["ingredient"]].drop_duplicates().copy()
print(f"unique ingredients: {len(ingredients)}")
print(f"recipe / ingredient links: {len(recipe_to_ingredient)}")

# convert relation to ids
ingredients["id"] = ingredients.index.copy()
recipe_to_ingredient = recipe_to_ingredient.merge(ingredients, on="ingredient")
recipe_to_ingredient = recipe_to_ingredient.rename(columns={"id": "ingredient_id"})
recipe_to_ingredient = recipe_to_ingredient.drop(columns=["ingredient"])
ingredients = ingredients.rename(columns={"ingredient": "name"})
recipes_df = recipes_df.drop(columns=["ingredients"])


# 3. Parse tags
recipe_to_tag = list()

for id, row in recipes_df.iterrows():
    tags = [t.strip().lower() for t in row["tags"].split(",")]
    for tag in tags:
        recipe_to_tag.append({"tag": tag, "recipe_id": id})

recipe_to_tag = pd.DataFrame(recipe_to_tag)
tags = recipe_to_tag[["tag"]].drop_duplicates().copy()
print(f"unique tags: {len(tag)}")
print(f"recipe / tag links: {len(recipe_to_tag)}")

# convert relation to ids
tags["id"] = tags.index.copy()
recipe_to_tag = recipe_to_tag.merge(tags, on="tag")
recipe_to_tag = recipe_to_tag.rename(columns={"id": "tag_id"})
recipe_to_tag = recipe_to_tag.drop(columns=["tag"])
recipes_df = recipes_df.drop(columns=["tags"])


# 4. Export to sqlite

# Create SQLite connection
conn = sqlite3.connect("./data/database.db")

# Write recipes table with proper data type for ID
recipes_df.to_sql(
    "recipes",
    conn,
    if_exists="replace",
    index=False,
    dtype={"id": "INTEGER PRIMARY KEY"},
)

ingredients[["id", "name"]].to_sql(
    "ingredients",
    conn,
    if_exists="replace",
    index=False,
    dtype={"id": "INTEGER PRIMARY KEY"},
)

# 4. Create many-to-many relation table
recipe_to_ingredient.to_sql(
    "recipe_ingredients",
    conn,
    if_exists="replace",
    index=False,
    dtype={
        "recipe_id": "INTEGER",
        "ingredient_id": "INTEGER",
        "FOREIGN KEY (recipe_id)": "REFERENCES recipes(id)",
        "FOREIGN KEY (ingredient_id)": "REFERENCES ingredients(id)",
        "PRIMARY KEY": "(recipe_id, ingredient_id)",
    },
)

# Create indexes for better performance
cursor = conn.cursor()
cursor.execute(
    """
        CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_recipe_id
        ON recipe_ingredients(recipe_id)
    """
)
cursor.execute(
    """
        CREATE INDEX IF NOT EXISTS idx_recipe_ingredients_ingredient_id
        ON recipe_ingredients(ingredient_id)
    """
)

# Close connection
conn.close()

print("Database created successfully!")
