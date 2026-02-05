"""Structure Collection + Database"""
# Imports
from utils.lab1 import calculate_doc_size, compute_sharding_metrics, convert_bytes_to_gb, compute_collection_volume

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
    def __init__(self, name, schema, stats, count_rule="direct"):
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
        
        # compute the number of documents in a collections
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
        self.sharding_analysis[strategy_name] = metrics
        return metrics

class Database:
    """
    Represents Databases as collections of Collections
    """
    def __init__(self, name):
        self.name = name
        self.collections = {}
        
    def add_collection(self, collection_obj):
        """
        Adds a Collection object to the database.
        """
        self.collections[collection_obj.name] = collection_obj
        
    def get_total_size_gb(self):
        """
        Calculates the sum of all collection sizes in GB.
        """
        return sum(c.total_size_gb for c in self.collections.values())

    def print_report(self):
        """
        Generates and prints the formatted size report for the database.
        Matches the format requested in the exercise.
        """
        print(f"--- REPORT FOR {self.name} ---")
        
        # Iterate over all collections to print details
        for name, col in self.collections.items():
            print(f"Collection: {name:<12} | Doc Size: {col.doc_size_bytes:>8,.0f} B | Count: {col.count:>12,.0f} | Total: {col.total_size_gb:>8.2f} GB")
            
        # Print total database size
        print(f">>>> TOTAL SIZE: {self.get_total_size_gb():.2f} GB\n")
