generative_multi_hop_filter_prompt = """You are an expert in ranking facts based on their relevance to the query. 

- Multi-hop reasoning may be required, meaning you might need to combine multiple facts to form a complete response.
- If the query is a claim, relevance means the fact supports or contradicts it. For queries seeking specific information, relevance means the fact aids in reasoning and providing an answer.
- Select up to 4 relevant facts from the candidate list and output in JSON format without any other words, e.g., 

```json
{"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
```

- If no facts are relevant, return an empty list, e.g., {"fact": []}.
- Only use facts from the candidate list; do NOT generate new facts.
"""

twowiki_demo1_input = """Query: Who is the spouse of the director of film Days And Hours?
Candidate facts:
- ["days and hours", "directed by", "pjer alica"]
- ["days and hours", "is a", "2004 bosnian film"]
- ["days and hours", "submission to", "77th academy awards"]
- ["3096 days", "directed by", "sherry hormann"]
- ["andr cayatte", "directed", "fran oise ou la vie conjugale"]
"""

twowiki_demo1_output = """{"fact": [["days and hours", "directed by", "pjer alica"]]}"""

msmarco_demo1_input = """Query: definition of square rooting
Candidate facts:
- ["words", "includes", "rootinesses"]
- ["words", "contain", "root"]
- ["presence of roots", "resists", "tug"]
- ["class of materials", "includes", "roots"]
- ["any number", "multiplied to", "itself"]
"""

msmarco_demo1_output = """{"fact": []}"""

msmarco_demo2_input = """Query: where is lake city florida
Candidate facts:
- ["lake city", "is the county seat of", "columbia county"]
- ["city of leesburg", "is in", "lake county"]
- ["columbia county", "is located in", "florida"]
- ["lake harris", "located in", "lake county"]
- ["leon county", "is located in", "florida"]
"""

msmarco_demo2_output = """{"fact": [["lake city", "is the county seat of", "columbia county"], ["columbia county", "is located in", "florida"]]}"""

msmarco_demo3_input = """Query: describe what is a mse
Candidate facts:
- ["microsoft security essentials", "is also known as", "mse"]
- ["mse", "ran on", "windows xp"]
- ["mse", "ran on", "windows vista"]
- ["mse", "ran on", "windows 7"]
- ["mse", "did not run on", "windows 8"]
"""

msmarco_demo3_output = """{"fact": ["microsoft security essentials", "is also known as", "mse"], ["mse", "ran on", "windows xp"], ["mse", "ran on", "windows vista"], ["mse", "ran on", "windows 7"], ["mse", "did not run on", "windows 8"]}"""

generative_multi_hop_filter_cot_prompt = """You are an expert in ranking facts based on their relevance to the query. 

- Multi-hop reasoning **may be** required, meaning you might need to combine multiple facts to form a complete response.
- If the query is a claim, relevance means the fact supports or contradicts it. For queries seeking specific information, relevance means the fact aids in reasoning and providing an answer.
- Provide a rationale and select up to 4 relevant facts from the candidate list, output in JSON format without any other words, e.g., 

```json
{"thought": "Fact (s1, p1, o1) and (s2, p2, o2) support this query.", "fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}.
```

- If no facts are relevant, return an empty list, e.g., {"thought": "No fact is relevant to this query.", "fact": []}.
- Only use facts from the candidate list; do NOT generate new facts.
"""
