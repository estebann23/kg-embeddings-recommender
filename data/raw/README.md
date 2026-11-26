# HUMMUS data set

Dataset as published in [1].

## Important note on use

- The data was collected from the website [food.com](https://www.food.com/) and may only be used for educational or
  scientific purposes.
- The code is licensed under [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html).

-------------------

## Access the data

The data is located in the [`data`](data) directory.

- The subdirectory [`crawler_init`](data/crawler_init) includes a file extracted from the FoodKG which lists all recipe
  sources from [food.com](https://www.food.com/).
- The subdirectory [`food_kg`](data/food_kg/) includes the simplified \& extended version of the FoodKG [2]
- The subdirectory [`hummus_data`](data/hummus_data/) includes the collected data as described in [1].
  Both, [raw](data/hummus_data/raw/) and [preprocessed](data/hummus_data/preprocessed/) data are available.
- The subdirectory [`hummus_graph`](data/hummus_graph/) includes the complete RDF graph.

To explore the HUMMUS knowledge graph, we recommend to use the [GraphDB](https://graphdb.ontotext.com/) tool. All 
".ttl"-files can be imported with the "GraphDB Server Import feature".

-------------------

## Description of the raw data crawled from [food.com](https://www.food.com/)

These data are available in [`data/food_com/raw/`](data/hummus_data/raw/).

Each feature is not cleaned and of format string. There are also empty or invalid cells.

### `members.csv`

Contains raw data describing each member (user), including the following features:

- `member_id`: A unique ID identifying each member.
- `member_url`: An url pointing to the members' food.com profile site.
- `member_name`: The members choosen name. If the name includes first and last name, the last name is abbreviated.
- `member_description`: A description given by the member himself.
- `member_status`: The members' status.
- `member_avg_rating`: An average value of all rating given by the member, between 1 and 5.
- `member_map_pin`: A location description, ranging from various scopes from country to state.
- `member_joined`: The date the member created their account.
- `follows_count`: The number of follows.
- `follow_me_count`: The number of followers.
- `avatar_url`: An url pointing to the members profile picture.

### `recipes.csv`

Contains raw data describing each recipe, including the following features:

- `recipe_id`: An unique ID identifying each recipe.
- `recipe_url`: An url pointing to the recipes' food.com site.
- `input_db_index`: This is a technical field expressing the index (if present) of the input-table from the FoodKG which
  served as a starting point.
- `food_kg_locator`: The corresponding nodes' identifier/IRI in the FoodKG.
- `food_com_unique_id`: An unique ID identifying each recipe, slightly differently formatted as the recipe_id.
- `submit_date`: The date the recipe was submitted.
- `last_changed_date`: The date the recipe was changed the last time.
- `title`: The title of the recipe.
- `author_id`: The member id of the recipes' author.
- `rating`: The average rating of the recipe.
- `description`: The description of the recipe.
- `duration`: The duration necessary to cook the recipe.
- `serves`: The number of servings of the recipe.
- `units`: The type of units given for the ingredients. Our data set only consists of the US measures.
- `ingredients`: A json-object including all recipes and possibly their category.
- `nutrition`: A json-object including all nutrition information.
- `directions`:  A json-object including all steps to take to cook the recipe.

### `reviews.csv`

Contains raw data describing each review, including the following features:

- `review_id`: The unique id of the review.
- `member_id`: The unique id of the member who wrote the review.
- `recipe_id`: The unique id of the recipe the review is written for.
- `review_url`: The url pointing to the reviews' page.
- `member_url`: The url pointing to the members' page.
- `recipe_url`: The url pointing to the recipes' page.
- `review_unique_id`: An unique ID identifying each review, slightly differently formatted as the review_id.
- `rating`: The rating of the review. The number is between 0 to 5. However, 0 refers to review without rating only.
- `text`: The text of the review.
- `likes`: The recieved likes of the review.
- `submit_date`: The submit date of the review.
- `last_modified_date`: The date when the review was changed the last time.

-------------------

## Description of the preprocessed data

These data are available in [`data/food_com/preprocessed`](data/hummus_data/preprocessed/).

We also provide pre-processed data with no Threshold Discarding, including some additional food.com tags, and nutrition
scores. We cleaned each feature, e.g. parsing the jsons, formatting the dates, changing the datatypes of numbers
previously represented as strings, etc. ...

If some adjustments are necessary, we provide the cleaning/import code in this repository.

In the following, we only name the additional features in comparison to the raw data.

### `members.csv` (`pp_members.csv`)

- `new_member_id`: A new unique id representing each member in the data set beginning from 0.

### `recipes.csv` (`pp_recipes.csv`)

- `recipe_url`: The url pointing to the original recipes' page.
- `average_rating`: The average rating of the recipe.
- `number_of_ratings`: The number of ratings of the recipe.
- `tags`: A list of tags for the current recipes collected by [3] from food.com.
- `servingSize [g]`: Extracted from the `nutrition` feature. Represents the size of one serving in g.
- `servingsPerRecipe`: Extracted from the `nutrition` feature. A number describing the number of servings when cooking
  the recipe as instructed.
- `calories [cal]`: Extracted from the `nutrition` feature. The calories of one serving in cal.
- `caloriesFromFat [cal]`: Extracted from the `nutrition` feature. The calories from fat of one serving in cal.
- `totalFat [g]`: Extracted from the `nutrition` feature. The total fat of one serving in g.
- `saturatedFat [g]`: Extracted from the `nutrition` feature. The saturated fat of one serving in g.
- `cholesterol [mg]`: Extracted from the `nutrition` feature. The cholesterol of one serving in mg.
- `sodium [mg]`: Extracted from the `nutrition` feature. The sodium/salt of one serving in mg.
- `totalCarbohydrate [g]`: Extracted from the `nutrition` feature. The total carbohydrates of one serving in g.
- `dietaryFiber [g]`: Extracted from the `nutrition` feature. The dietary fibers of one serving in g.
- `sugars [g]`: Extracted from the `nutrition` feature. The sugar of one serving in g.
- `protein [g]`: Extracted from the `nutrition` feature. The protein of one serving in g.
- `direction_size`: The number of steps needed to cook the recipe.
- `ingredients_sizes`: The number of ingredients necessary for the recipe.
- `new_author_id`: A new unique id representing each member in the data set beginning from 0.
- `new_recipe_id`: A new unique id representing each recipe in the data set beginning from 0.
- `who_score`: A recipe healthiness score based on WHO methodology. The range is 0-14, with 14 as the best. For more information refer to the next section "Nutrition scores".
- `fsa_score`: A recipe healthiness score based on the UK Food Standards Agency (FSA) nutrient profiling system. The range is 0-8, with 8 as the best. For more information refer to the next section "Nutrition scores".
- `nutri_score`: A recipe nutri-score. A is the best result, E is the worst result. For more information refer to the next section "Nutrition scores".
- `normalization_comment`: Indicator of the required normalisation according to the following heuristic:  
  A) Make sure that all ingredients weight less than the serving size (because this is, according to food.com, the
  scale) --> normalize to 100g. In this case, no value is assigned.
  B) The serving size is faulty. Test if the ingredients summed weight is less than `serving size` *
  `servings per recipe`. --> normalize based on that value & mark the value with one star '*'.
  C) The serving size does not correspond to both alternatives. --> skip normalization & mark the value with two
  stars '**'.
- `ingredient_food_kg_urls`: A list of urls pointing to the food_kg ingredients' urls.
- `ingredient_food_kg_names`: A list of ingredient names derived by the food_kg mapped ingredients.

### `reviews.csv` (`pp_reviews.csv`)

- `new_member_id`: A new unique id representing each member in the data set beginning from 0.
- `new_recipe_id`: A new unique id representing each recipe in the data set beginning from 0.

In addition to the new ids, we also modified the following user-recipe interactions and added those into the set of
reviews. Also this procedure can be changed/modified when calling our data cleaning code.

- We added authorship relations as reviews with a rating of 6. The idea is, when only using the reviews as collaborative
  feature, that a recipes' author normally might rate his own creating the best.
- We modified the ratings of reviews or comments without ratings from 0 to 2.5. The idea is, when only using the reviews
  as collaborative feature, that those comments/reviews also symbolises a certain interest, even without a rating.
  However, a rating of 0 would be interpreted as the worst possible rating, so we decided to set this number to an
  average rating.

-------------------

## Nutrition scores

In this section we briefly describe the construction and modifications made to the nutrition scores in order to measure
the "healthiness" of one recipe. 

Keep in mind, that some recommendations tackle additional micro- and macronutrients (i.e. vitamins) which are however not in our dataset and thus are discarded.

Also note, that all scores can alternatively be calculated to result a normalized scale from 0 to 1 instead of their own custom scales.

### WHO score

This score is based on WHO methodology, which expresses dietary reference values for a range of different micro- and
macro nutrients.
It is calculated as a sum of individual scores for the following macronutrients:
Protein, total carbohydrate, sugars, total fat, saturated fat, dietary fiber, and sodium.

For each macronutrient, the thresholds are defined by WHO are as follows:

| Macronutrient thresholds | Protein \[g]<br/>per 100 g | Total Carbohydrate \[g]<br/>per 100 g | Sugars \[g]<br/>per 100 g | Total fat \[g]<br/>per 100 g | Saturated fat \[g]<br/>per 100 g | Dietary fiber<br/>per 100 g | Sodium \[g]<br/>per serving |
|:------------------------:|:--------------------------:|:-------------------------------------:|:-------------------------:|:----------------------------:|:--------------------------------:|:---------------------------:|:---------------------------:|
|     lower bound (lb)     |             10             |                  55                   |             0             |              15              |                0                 |              0              |              0              |
|     upper bound (ub)     |             15             |                  75                   |            10             |              30              |                10                |              3              |              2              |
|      minimize (min)      |           False            |                 False                 |           True            |             True             |               True               |            False            |            True             |

The overall score is the sum of the scores of each macronutrient.

<img alt="who_score" height="25" src="equations/who_score.png"/>

Note that sodium is a special case, since it will be normalized to the serving size instead of 100g.
Depending on if a given macronutrient is to be minimized (i.e. sugars are unhealthy and thus must be minimized) or
maximized (i.e. dietary fiber is healthy and thus must be maximized), the score is calculated as follows:

<img alt="who minimize" height="80" src="equations/who_minimize.png"/>

<img alt="who maximize" height="80" src="equations/who_maximize.png"/>


Thus, the overall WHO score scale ranges from 0 (unhealthy) to 14 (healthy).

For more details, you can
check [WHO healthy diet key facts](https://www.who.int/news-room/fact-sheets/detail/healthy-diet).

### FSA score

This score is based on the UK Food Standards Agency (FSA) nutrient profiling system.
This system expresses dietary reference values for a range of different macronutrients:
Total fat, saturated fat, sugars, and sodium.

For each macronutrient, the thresholds defined by the FSA are as follows:

|    Macronutrient thresholds    | Total fat \[g] | Saturated fat \[g] | Sugars \[g] | Sodium \[g] |
|:------------------------------:|:--------------:|:------------------:|:-----------:|:-----------:|
| lower bound (lb)<br/>per 100 g |       3        |        1.5         |      5      |     0.3     |
| upper bound (ub)<br/>per 100 g |      17.5      |         5          |    22.5     |     1.5     |
| upper bound per portion (ubp)  |       21       |        0.3         |     1.8     |     1.8     |

The overall score is the sum of the scores of each macronutrient.

<img alt="fsa_score" height="25" src="equations/fsa_score.png"/>

Each Macronutrient score is calculated as follows:

<img alt="fsa" height="80" src="equations/fsa.png"/>



Thus, the overall FSA score scale ranges from 0 (unhealthy) to 8 (healthy).

For more details, refer
to [Guide to creating a front of pack (FoP) nutrition label for pre-packed products sold through retail outlets](https://www.food.gov.uk/sites/default/files/media/document/fop-guidance_0.pdf).

### NutriScore

The NutriScore is directly developed to measure the healthiness of certain food products.
It has a five-colour nutritional rating system that assigns products a rating letter from A (best) to E (worst).
Originally, it was derived from the FSA score.
Here, we implement a simplified version that takes into account the following parameters:
Energy density (calories), saturated fats, sugar, sodium, fiber, and protein.

As the NutriScore is a bit more complicated, we only give a brief overview of the algorithm in the following figure.

<img alt="nutri score" height="600" src="equations/nutri_score.png"/>


Note, that for the NutriScore in the current preprocessed dataset, the parts of "fruits, vegetables", "cheese", and "beverages" are missing. 
Thus, it underestimates very healthy recipes (with a lot of fruits or vegetables) and those including cheese.
Also, beverages are overestimated in terms of healthiness.
However, we provide both implementations (simplified and complete) of the NutriScore in the code.

All in all, the overall NutriScore scale ranges from A (healthy) to E (unhealthy).

For more details, you can refer
to [The Nutri-Score: A Science-Based Front-of-Pack Nutrition Label](https://www.iarc.who.int/wp-content/uploads/2021/09/IARC_Evidence_Summary_Brief_2.pdf)
and [Update of the Nutri-Score algorithm 2022](https://www.bmel.de/SharedDocs/Downloads/EN/_Food-and-Nutrition/nutri-score-update-algorithm.pdf?__blob=publicationFile&v=2).

Let's depict the distribution of the recipes based on the calculated nutri-score values:

<img alt="Distribution of all recipes per nutri-score" height="300" src="img/all-recipes-per-nutriscore.png"/>

It can be noted that most of the recipes are scored D or even E.

The same tendency can be seen for rated recipes (i.e. having at least one rating):

<img alt="Distribution of rated recipes per nutri-score" height="300" src="img/rated-recipes-per-nutriscore.png"/>

When looking at the distribution of nutri-score values and ratings together no clear correlation can be seen.

<img alt="Distribution of recipe ratings per nutri-score" height="300" src="img/avg_distribution_nutri_ratings.jpg"/>

-------------------

## Code

The code and useful scripts are located in the following directories. Make sure to unpack/download the required data in
the responding directories.

### Crawler [`code/crawler`](code/crawler/)

All code related to the crawler used for extracting recipes from [food.com](https://www.food.com/). It
uses [`scrapy`](https://scrapy.org/) as main framework.

It takes as input the file `data/crawler_init/recipe_sourcesAndDomains.csv`

### Data Science related [`code/data_science`](code/data_science/)

Includes code for cleaning, exploration, simplification \& extension of the `FoodKG`.

As an input, the following files are required for loading the crawled (raw) data:

- [`data/food_com/raw/recipes.csv`](data/hummus_data/raw/recipes.zip),
- [`data/food_com/raw/reviews.csv`](data/hummus_data/raw/reviews.zip),
- [`data/food_com/raw/members.csv`](data/hummus_data/raw/members.zip)

They should be unziped to [`data/food_com/`](data/hummus_data/). Alternatively, you can modify the data
location (`data_location = '../../data/food_com/'`)
in [`code/data_science/dataset_scripts.ipynb`](code/data_science/dataset_scripts.ipynb).

#### FoodKG simplification [`code/data_science/food_KG_simplification/`](code/data_science/food_KG_simplification/)

Simplifies the FoodKG [2] by removing and merging a lot of nodes.
Keeps FoodOn Data and recipe Data for traversal.
Extends the graph by user-item (in our case user-recipe) interactions and some metadata.
For more details refer to the code.

#### Data cleaning, exploration & experiments

Scripts located in this directory are for data cleaning and show a basic usage with the data.

As input, the following files are optional:

- [Optional] For merging an additional food.com data set [3] in the current one (in order to gain food tags):
  `data/food_com/RAW_recipes.csv` (This dataset is not included in this repository, download it
  from https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)
- [Optional] For merging food product data as labels into the dataframes directly:
  `data/food_kg/graph_ingredient_labels.csv`
- [Optional] For merging food ingredients data into the dataframes directly:
  `data/food_kg/graph_ingredient_mapping.csv`

#### Experiments including post-processing

Scripts located in this directory are the final step, logically following the cleaning part and the first few
experiments. They include a BPR-recommender including hyperparameter optimizations and three different kinds of healthy
post-processing methods:

- Filter by threshold: Increases the healthiness by removing recommended recipes below a certain threshold.
- Filter by percentage: Increases the healthiness by removing the x% unhealthiest recommended recipes of each user.
- Replace unhealthy recipes: Finds for each recipe content-wise similar ones. Replaces recommended recipes if there is a
  fitting other recipe (similar and healthy enough).

All methods can be processed on any nutrition score. There is a trade-off between maximizing healthiness and accuracy.

-------------------

## Results & Visualizations

All results and/or insights which are found here and/or in the paper can be replicated by the provided code.

- Density of nutrient values in the data set after normalization:

<img alt="Density nutrient values" height="200" src="img/nutrientDensity.png"/>

- The WHO score, FSA score, and NutriScore rating the healthiness of each recipe in the HUMMUS data set. For the
  NutriScore,
  the colour encoding follows its specifications. For the FSA score, the colour encoding is a similar split between the
  colour shades red,
  orange, and green. The WHO score is not evenly coloured but follows the colours of the distributions of the
  NutriScore.

<img alt="Scores" height="400" src="img/scores.png"/>

- Exemplary structure of the simplified FoodKG; structure-wise irrelevant relations are discarded:

<img alt="Simplified FoodKG" height="200" src="img/simplified_FoodKG.png"/>

- SPARQL query to extract 100 vegetarian recipes from the simplified FoodKG:

<img alt="SPARQL Query" height="200" src="img/sparqlQuery.png"/>

- Heatmap of a subset of features of the data set. The data behaves as expected: Carbohydrates, fats and calories are
  correlated.

<img alt="Heatmap" height="500" src="img/recipeHeatmap.png"/>

- Data set size and sparsity after threshold discarding with different parameters 𝑘 = (𝑘_users, 𝑘_recipes):

<img alt="Data set information" height="150" src="img/hummusDataSetInformation.png"/>

- Resulting scores for our data set with a sparsity of 99.89351% of the processed models from different python
  libraries (implicit, irec, recommenders):

<img alt="Recommendation resulting scores" height="400" src="img/resultingScores.png"/>

- Resulting scores for the MovieLens100k data set with a sparsity of 93.69%. The results are computed by the team of the
  recommenders library:

<img alt="Comparison MovieLens100k scores" height="180" src="img/comparisonMovieLens.png"/>

-------------------

# References

[1] Felix Bölz, Diana Nurbakova, Sylvie Calabretto, Armin Gerl, Lionel Brunie, and Harald Kosch. 2023. HUMMUS: A Linked,
Healthiness-Aware, User-centered and Argument-Enabling Recipe Data Set for Recommendation. In Seventeenth ACM Conference
on Recommender Systems (RecSys '23), September 18--22, 2023, Singapore, Singapore. ACM, New York, NY, USA 11 Pages.
https://doi.org/10.1145/3604915.3609491

[2] Haussmann, S., Seneviratne, O., Chen, Y., Ne’eman, Y., Codella, J., Chen, C. H., … Zaki, M. J. (2019). FoodKG: A
Semantics-Driven Knowledge Graph for Food Recommendation. In Lecture Notes in Computer Science (including subseries
Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics) (Vol. 11779 LNCS, pp. 146–162).
Springer. https://doi.org/10.1007/978-3-030-30796-7_10

[3] Majumder, B. P., Li, S., Ni, J., & McAuley, J. (2019). Generating personalized recipes from historical user
preferences. In EMNLP-IJCNLP 2019 - 2019 Conference on Empirical Methods in Natural Language Processing and 9th
International Joint Conference on Natural Language Processing, Proceedings of the Conference (pp. 5976–5982).
Association for Computational Linguistics. https://doi.org/10.18653/v1/d19-1613


