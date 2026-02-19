"""Class Database"""
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