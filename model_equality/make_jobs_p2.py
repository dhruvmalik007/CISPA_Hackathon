"""Writes jobs_p2.jsonl: 40 long free-text prompts (about 200-word answers) for the local likelihood test.
Same file for practice and scored endpoints; collect.py --cap controls how much scored budget is used."""
import json

P = [
    # encyclopedic continuations
    "Continue this encyclopedia article for about 200 words: 'The Roman aqueducts were a network of channels that'",
    "Continue this encyclopedia article for about 200 words: 'Photosynthesis is the process by which green plants'",
    "Continue this encyclopedia article for about 200 words: 'The Silk Road was an ancient network of trade routes that'",
    "Continue this encyclopedia article for about 200 words: 'Volcanoes form when molten rock from deep inside the Earth'",
    "Continue this encyclopedia article for about 200 words: 'The printing press, developed by Johannes Gutenberg around 1440,'",
    "Continue this encyclopedia article for about 200 words: 'Honeybees live in colonies that are organised around a single queen,'",
    "Continue this encyclopedia article for about 200 words: 'The French Revolution began in 1789, when'",
    "Continue this encyclopedia article for about 200 words: 'Black holes are regions of spacetime where gravity is so strong that'",
    # explanations
    "Explain in about 200 words how vaccines train the immune system.",
    "Explain in about 200 words why the sky is blue and sunsets are red.",
    "Explain in about 200 words how a refrigerator keeps food cold.",
    "Explain in about 200 words what inflation is and why central banks raise interest rates.",
    "Explain in about 200 words how GPS works.",
    "Explain in about 200 words the difference between weather and climate.",
    "Explain in about 200 words how compound interest works, with a simple example.",
    "Explain in about 200 words how a bill becomes a law in a parliamentary democracy.",
    # stories
    "Write a short story (about 200 words) about a robot who learns to paint.",
    "Write a short story (about 200 words) about two strangers sharing an umbrella in the rain.",
    "Write a short story (about 200 words) set in a bakery at 4 a.m.",
    "Write a short story (about 200 words) about a lost dog finding its way home.",
    "Write a short story (about 200 words) about a girl who discovers a door in her garden.",
    "Write a short story (about 200 words) about the last day of summer camp.",
    # practical
    "Give step-by-step instructions for making a simple tomato pasta sauce.",
    "Give ten practical tips for someone starting to run.",
    "Write a polite email to a landlord asking for a broken heater to be repaired.",
    "Describe a perfect three-day trip to Lisbon.",
    "Write a product description (about 150 words) for a reusable water bottle.",
    "List the pros and cons of working from home.",
    # code
    "Write a Python function that checks whether a string is a palindrome, with comments and two test cases.",
    "Write a Python script that reads a CSV file and prints the average of a column named 'price'.",
    "Write a JavaScript function that debounces another function, and explain how it works.",
    "Write a SQL query that finds the top 5 customers by total order value, and explain each clause.",
    # other languages
    "Écris un texte d'environ 150 mots sur l'histoire de la tour Eiffel.",
    "Escribe un texto de unas 150 palabras sobre la importancia del agua.",
    "Schreibe einen Text von etwa 150 Wörtern über den Herbst im Wald.",
    "Scrivi un testo di circa 150 parole su una giornata al mare.",
    # misc
    "Summarise the plot of a classic fairy tale of your choice in about 200 words.",
    "Write a motivational speech (about 200 words) for a team before an important match.",
    "Describe the water cycle to a ten-year-old in about 200 words.",
    "Write a dialogue (about 200 words) between a customer and a bookshop owner.",
]
assert len(P) == 40
with open("jobs_p2.jsonl", "w", encoding="utf-8") as f:
    for i, p in enumerate(P):
        f.write(json.dumps({"id": f"ft{i:02d}", "prompt": p, "repeats": 1}, ensure_ascii=False) + "\n")
print("jobs_p2.jsonl: 40 prompts, 1 answer each")