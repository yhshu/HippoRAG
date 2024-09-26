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

input_demo1 = """Query: Who is the spouse of the director of film Days And Hours?
Candidate facts:
- ["days and hours", "directed by", "pjer alica"]
- ["days and hours", "is a", "2004 bosnian film"]
- ["days and hours", "submission to", "77th academy awards"]
- ["3096 days", "directed by", "sherry hormann"]
- ["andr cayatte", "directed", "fran oise ou la vie conjugale"]
"""

output_demo1 = """{"fact": [["days and hours", "directed by", "pjer alica"]]}"""

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
