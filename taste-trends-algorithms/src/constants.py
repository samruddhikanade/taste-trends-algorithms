"""Data rules shared by Spark preprocessing and the pure-Python tests."""

MIN_RATING, MAX_RATING = 0.5, 5.0          # MovieLens uses a 0.5-5.0 half-star scale
NO_GENRE = "(no genres listed)"            # MovieLens placeholder for movies without genres
YEAR_PATTERN = r"\((\d{4})\)\s*$"          # release year at the end of a title, e.g. "Heat (1995)"
