"""Class Collection"""
# Imports
from utils.help_lab1 import calculate_doc_size, compute_sharding_metrics, convert_bytes_to_gb, compute_collection_volume

# Config
SIZES = {
    "integer": 8,
    "number": 8,        # Price, VAT
    "string": 80,       # Standard string
    "date": 20,         # Specific format
    "long_string": 200, # Desc, URL, Comment, Address
    "overhead": 12      # Key+Value overhead
}

LONG_STRINGS = ["description", "image_url", "comment", "address", "headOffice"]
DATES = ["date", "birthDate", "deliveryDate"]

class Collection:
    def __init__(self, name, schema, stats = {}, count_rule="direct"):
        self.name = name
        self.schema = schema
        self.stats = stats 
        self.sharding_analysis = {}

        # Calcul of bytes
        self.doc_size_bytes = calculate_doc_size(
            schema=self.schema,
            sizes_config=SIZES,          
            stats_config=self.stats,     
            dates_list=DATES,           
            long_strings_list=LONG_STRINGS 
        )
        
        # compute the number of documents in a collection
        self.count = compute_collection_volume(self.name, self.stats, count_rule)
            
        # Total size in GB
        self.total_size_gb = convert_bytes_to_gb(self.doc_size_bytes, self.count)
        

    def analyze_sharding(self, key_stat_name):
        """
        Analyzes distribution for a given key.
        """
        key_cardinality = self.stats.get(key_stat_name, 0)
        metrics = compute_sharding_metrics(self.count, key_cardinality)
        strategy_name = f"{self.name} - #{key_stat_name}"
        metrics['strategy'] = strategy_name
        self.sharding_analysis[strategy_name] = metrics
        return metrics

    def get_projected_doc_size(self, projected_keys):
        """
        Compute the size of a reducted document obtained by keeping the keys : projected_key
        """
        if not projected_keys or projected_keys == ["*"]:
            return self.doc_size_bytes
        
        # Building a temporary schema with asked keys
        mini_schema = {"type": "object", "properties": {}}

        original_props = self.schema.get("properties", {})
        for key in projected_keys:
            if key in original_props:
                # Copyig the properties we are looking for in the mini_schema
                mini_schema["properties"][key] = original_props[key]

        # Computing the size of the mini shema
        return calculate_doc_size(
            mini_schema, 
            SIZES,
            self.stats,
            DATES, 
            LONG_STRINGS

        )
    