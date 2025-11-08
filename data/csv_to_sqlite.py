import pandas as pd
import sqlite3
import json

# Read CSV file
df = pd.read_csv("./data/recipes_data.csv", nrows=20000)
print(f"loaded {len(df)} recipes")

# 1. Convert directions to string
df["directions"] = df["directions"].apply(
    lambda x: "\n".join(json.loads(x)) if pd.notna(x) else ""
)

clean_data = []  # Use a list to collect data, then create DataFrame at the end
ingredients_data = set()

for _, row in df.iterrows():
    try:
        # Parse the NER column which contains ingredient names
        ner_names = json.loads(row["NER"]) if pd.notna(row["NER"]) else []
        raw_ingredient_names = (
            json.loads(row["ingredients"]) if pd.notna(row["ingredients"]) else []
        )
        ingredient_names = [
            n for n in ner_names if any((r for r in raw_ingredient_names if n in r))
        ]
        all_ingredients_parsed = len(ingredient_names) == len(raw_ingredient_names)
        if ingredient_names == [] or not all_ingredients_parsed:
            continue

        ingredients_data.update(ingredient_names)
        clean_data.append(
            {
                "title": row["title"],
                "directions": row["directions"],
                "link": row["link"],
                "ingredients": ingredient_names,
            }
        )
    except Exception as e:
        print(e)

# Add auto-increment ID column
recipes_df = pd.DataFrame(clean_data)
recipes_df["id"] = range(1, len(recipes_df) + 1)
print(f"clean recipes: {len(recipes_df)} / {len(df)}")

# Create ingredients dataframe
ingredients_df = pd.DataFrame({"name": list(ingredients_data)})
ingredients_df["id"] = range(1, len(ingredients_df) + 1)
print(f"unique ingredients: {len(ingredients_df)}")

# 2. Extract ingredients from NER to a separate dataframe
recipe_ingredients_data = []
for _, row in recipes_df.iterrows():
    for ingredient_name in row["ingredients"]:
        # Add to recipe_ingredients junction table
        recipe_ingredients_data.append(
            {"recipe_id": row["id"], "ingredient_name": ingredient_name}
        )
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
print(f"Recipe-Ingredient relationships: {len(recipe_ingredients_df)}")


# 3. Export to sqlite

# Create SQLite connection
conn = sqlite3.connect("./data/database.db")

# Write recipes table with proper data type for ID
recipes_df[["id", "title", "directions", "link"]].to_sql(
    "recipes",
    conn,
    if_exists="replace",
    index=False,
    dtype={"id": "INTEGER PRIMARY KEY"},
)

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
