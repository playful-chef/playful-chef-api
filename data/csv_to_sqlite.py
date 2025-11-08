import pandas as pd
import sqlite3
import json

# Read CSV file
df = pd.read_csv("./data/recipes_data.csv")
print(f"total {len(df)} recipes")
df = df[:10000]
print(f"picked {len(df)} recipes")

# Add auto-increment ID column
df["id"] = range(1, len(df) + 1)

# 1. Convert directions to string
df["directions"] = df["directions"].apply(
    lambda x: "\n".join(json.loads(x)) if pd.notna(x) else ""
)

# Create SQLite connection
conn = sqlite3.connect("./data/database.db")

# Write recipes table with proper data type for ID
recipes_df = df[["id", "title", "directions", "link"]].copy()
recipes_df.to_sql(
    "recipes",
    conn,
    if_exists="replace",
    index=False,
    dtype={"id": "INTEGER PRIMARY KEY"},
)

# 2. Extract ingredients from NER to a separate dataframe
ingredients_data = []
recipe_ingredients_data = []


for _, row in df.iterrows():
    recipe_id = row["id"]
    try:
        # Parse the NER column which contains ingredient names
        ner_names = json.loads(row["NER"]) if pd.notna(row["NER"]) else []
        raw_ingredient_names = (
            json.loads(row["ingredients"]) if pd.notna(row["ingredients"]) else []
        )
        ingredient_names = [
            n for n in ner_names if any((r for r in raw_ingredient_names if n in r))
        ]

        for ingredient_name in ingredient_names:
            # Add to ingredients list (we'll deduplicate later)
            ingredients_data.append({"name": ingredient_name})

            # Add to recipe_ingredients junction table
            recipe_ingredients_data.append(
                {"recipe_id": recipe_id, "ingredient_name": ingredient_name}
            )
    except (ValueError, SyntaxError):
        # Handle cases where NER column cannot be parsed
        continue

# Create ingredients dataframe and deduplicate
ingredients_df = pd.DataFrame(ingredients_data)
ingredients_df = ingredients_df.drop_duplicates(subset=["name"]).reset_index(drop=True)
ingredients_df["id"] = range(1, len(ingredients_df) + 1)

# Create recipe_ingredients junction dataframe
recipe_ingredients_df = pd.DataFrame(recipe_ingredients_data)

# Merge to get ingredient IDs in the junction table
recipe_ingredients_df = recipe_ingredients_df.merge(
    ingredients_df[["id", "name"]],
    left_on="ingredient_name",
    right_on="name",
    how="left",
)
recipe_ingredients_df = recipe_ingredients_df[["recipe_id", "id"]].rename(
    columns={"id": "ingredient_id"}
)

# 3. Export ingredients to separate table
ingredients_df[["id", "name"]].to_sql(
    "ingredients",
    conn,
    if_exists="replace",
    index=False,
    dtype={"id": "INTEGER PRIMARY KEY"},
)

# 4. Create many-to-many relation table
recipe_ingredients_df.to_sql(
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
print(f"Recipes: {len(recipes_df)}")
print(f"Ingredients: {len(ingredients_df)}")
print(f"Recipe-Ingredient relationships: {len(recipe_ingredients_df)}")
