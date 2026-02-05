"""Structure Collection + Database"""
# import

from utils.lab1 import calculate_doc_size, compute_sharding_metrics, convert_bytes_to_gb
# Config
# ----- CONFIG -----
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
    def __init__(self, name, schema, config_module, count_rule="direct"):
        self.name = name
        self.schema = schema
        # On stocke la config pour l'utiliser plus tard si besoin
        self.config = config_module 
        
        # --- ETAPE 1 : CALCUL DE LA TAILLE (Bytes) ---
        # On appelle la fonction utilitaire qui retourne un INT
        self.doc_size_bytes = calculate_doc_size(
            schema=self.schema,
            sizes_config=self.config.SIZES,
            stats_config=self.config.STATS,
            dates_list=self.config.DATES,
            long_strings_list=self.config.LONG_STRINGS
        )
        
        # --- ETAPE 2 : CALCUL DU VOLUME (Count) ---
        self.count = 0
        if count_rule == "direct":
            # Mapping dynamique via le nom
            # Astuce: on convertit "Product" en "nb_products" pour chercher dans stats
            stat_key = f"nb_{name.lower()}s" # ex: Product -> nb_products
            self.count = self.config.STATS.get(stat_key, 0)
            
            # Gestion des cas particuliers si le mapping automatique échoue
            if self.count == 0 and name == "OrderLine": 
                self.count = self.config.STATS["nb_orderlines"]
                
        elif count_rule == "product_x_warehouse":
            self.count = self.config.STATS["nb_products"] * self.config.STATS["nb_warehouses"]
            
        
        self.total_size_gb = convert_bytes_to_gb(self.doc_size_bytes,self.count)
        
        # On initialise un dictionnaire vide pour les futurs calculs de sharding
        self.sharding_analysis = {} 

    def analyze_sharding(self, key_stat_name):
        """
        Méthode qui utilise la fonction utilitaire 'compute_sharding_metrics'
        et stocke le résultat dans l'attribut de l'objet.
        """
        key_cardinality = self.config.STATS.get(key_stat_name, 0)
        
        # Appel de la fonction Pure qui retourne le dictionnaire
        metrics = compute_sharding_metrics(self.count, key_cardinality)
        
        # On stocke le résultat dans l'objet
        strategy_name = f"{self.name} - #{key_stat_name}"
        self.sharding_analysis[strategy_name] = metrics
        
        return metrics # On retourne aussi pour affichage immédiat si besoin