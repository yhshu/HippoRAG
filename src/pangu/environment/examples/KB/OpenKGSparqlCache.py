import json
import logging
import os

from src.pangu.query_util import execute_query_with_virtuoso


class OpenKGSparqlCache:

    def __init__(self, cache_path: str = 'output/ttl_kg/openkg_sparql_cache.json'):
        self.cache_path = cache_path
        if os.path.isfile(cache_path):
            print(f"Loading cache from {cache_path}")
            self.load_cache(cache_path)
        else:
            print(f"Cache file {cache_path} not found, creating a new one")
            self.cache = {"types": {}, "in_relations": {}, "out_relations": {}, "in_entities": {},
                          "out_entities": {},
                          "cmp_entities": {},
                          "is_reachable": {},
                          "is_intersectant": {},
                          "sparql_execution": {}}

    def get_sparql_execution(self, sparql_query: str):
        try:
            if sparql_query not in self.cache["sparql_execution"]:
                rows = execute_query_with_virtuoso(sparql_query)
                self.cache["sparql_execution"][sparql_query] = rows
            return self.cache["sparql_execution"][sparql_query]
        except Exception as e:
            print(f"get_sparql_execution: {sparql_query}")
            print(e)
            return []

    def get_entity_in_relations(self, entity: str):
        if entity not in self.cache["in_relations"]:
            rows = execute_query_with_virtuoso(f"SELECT DISTINCT ?p WHERE {{ ?s ?p <{entity}> }}")
            self.cache["in_relations"][entity] = [row[0] for row in rows]
        return self.cache["in_relations"][entity]

    def get_literal_in_relations(self, literal: str):
        if literal not in self.cache["in_relations"]:
            rows = execute_query_with_virtuoso(f"SELECT DISTINCT ?p WHERE {{ ?s ?p \"{literal}\" }}")
            self.cache["in_relations"][literal] = [row[0] for row in rows]
        return self.cache["in_relations"][literal]

    def get_entity_out_relations(self, entity: str):
        if entity not in self.cache["out_relations"]:
            rows = execute_query_with_virtuoso(f"SELECT DISTINCT ?p WHERE {{ <{entity}> ?p ?o }}")
            self.cache["out_relations"][entity] = [row[0] for row in rows]
        return self.cache["out_relations"][entity]

    def load_cache(self, path: str):
        with open(path, 'r') as f:
            self.cache = json.load(f)

    def save_cache(self, path: str):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Use an absolute path for the temporary file
            temp_file_path = os.path.abspath(os.path.join(os.path.dirname(path), 'sparql_cache_temp.json'))

            # Write to the temporary file
            logging.info(f"Writing to temporary file: {temp_file_path}")
            with open(temp_file_path, 'w') as f:
                json.dump(self.cache, f)

            # Check if the temporary file was created successfully
            if os.path.isfile(temp_file_path):
                # Try to replace the existing file (if it exists) with the new one
                logging.info(f"Replacing {path} with {temp_file_path}")
                os.replace(temp_file_path, path)
                logging.info("Cache saved successfully")
            else:
                raise FileNotFoundError(f"Temporary file {temp_file_path} was not created")

        except Exception as e:
            logging.error(f"Error saving cache: {str(e)}")
            # If an error occurs, try to remove the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise
