""" 
Utils fonction to implement the structure asked in the lab1 
It is a adaptated copy of the practise work
"""

# ----- Utils Functions ------
def compute_collection_volume(collection_name, stats_config, rule):
    """
    Determines the number of documents in a collection based on stats and rules.
    """
    # CASE 1: Count the number of product
    # In the lab, it is said that we have for each product there nb_warehouses documents 
    if rule == "product_x_warehouse":
        n_prod = stats_config.get("nb_products", 0)
        n_warehouses = stats_config.get("nb_warehouses", 0)
        return n_prod * n_warehouses

    # CASE 2: Direct Mapping with the SIZE Config
    elif rule == "direct":
        # formatting the collection to tget the volume
        stat_key = f"nb_{collection_name.lower()}s"
        count = stats_config.get(stat_key, 0)
        
        # 2. Specific fix for OrderLine
        if count == 0 and collection_name == "OrderLine":
            return stats_config.get("nb_orderlines", 0)
            
        return count
        
    return 0

def get_field_size(field_name, field_type, sizes_config, dates_list, long_strings_list):
    """
    Return the size of a specific field in bytes based on its field_type
    """
    if field_type in ["integer", "number"]:
        return sizes_config["integer"]
    
    elif field_type == "string":
        if field_name in dates_list:
            return sizes_config["date"]
        elif field_name in long_strings_list:
            return sizes_config["long_string"]
        else:
            return sizes_config["string"]
            
    return 0

def calculate_doc_size(schema, sizes_config, stats_config, dates_list, long_strings_list, context_name=""):
    """
    Recursively calculates the size of a document (or sub-document) in bytes.
    """
    total_size = 0
    
    # CASE 1: Object (Dictionary) -> Recursive call on properties
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        for key, sub_schema in properties.items():
            # Calculate size of sub-field + Key Overhead (12B)
            field_size = calculate_doc_size(sub_schema, sizes_config, stats_config, dates_list, long_strings_list, context_name=key)
            total_size += (sizes_config["overhead"] + field_size)
            
    # CASE 2: Array (List) -> Recursive call on items
    elif schema.get("type") == "array":
        item_schema = schema.get("items")
        single_item_size = calculate_doc_size(item_schema, sizes_config, stats_config, dates_list, long_strings_list, context_name)
        
        # Determine array length based on context/statistics
        multiplier = 1
        if "categories" in context_name:
            multiplier = stats_config.get("avg_cat_per_prod", 1)
        elif "stocks" in context_name:
            multiplier = stats_config.get("nb_warehouses", 1) # Rule: 1 stock per warehouse
        elif "order_lines" in context_name:
            # Dynamic calculation: Total Lines / Total Products
            nb_ol = stats_config.get("nb_orderlines", 0)
            nb_prod = stats_config.get("nb_products", 1)
            multiplier = int(nb_ol / nb_prod) if nb_prod > 0 else 0
        
        # Size = Array Overhead + (Number of items * Item Size)
        total_size = sizes_config["overhead"] + (multiplier * single_item_size)
        
    # CASE 3: Primitive Value -> Get direct size
    else:
        field_type = schema.get("type")
        total_size = get_field_size(context_name, field_type, sizes_config, dates_list, long_strings_list)
        
    return total_size

def convert_bytes_to_gb(doc_size_bytes, collection_count):
    """Converts Bytes -> GB"""
    total_bytes = doc_size_bytes * collection_count
    return total_bytes / (1024**3)

def compute_sharding_metrics(collection_count, key_cardinality, nb_servers=1000):
    """
    Computes distribution statistics for sharding strategies
    """
    # Calculate averages per server
    avg_docs = collection_count / nb_servers
    avg_keys = key_cardinality / nb_servers
    
    # Return structured metrics with risk analysis (Skew/Empty servers)
    return {
        "nb_servers": nb_servers,
        "collection_count": collection_count,
        "key_cardinality": key_cardinality,
        "avg_docs_per_server": avg_docs,
        "avg_keys_per_server": avg_keys,
    }