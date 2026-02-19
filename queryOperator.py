
""" Class Query operator (lab2)"""
class QueryOperator:
    """
    Class to compute the cost
    """

    @staticmethod
    def op_filter(collection, filter_key, selectivity=0.01, projected_keys=None, sharding_key=None, nb_servers=1000):
        """
        Operator 1 & 2 : Filter (With or without sharding)

        Args:
        - collection (Collection object): the target collection to be queried.
        - filter_key (str):The attribute name used in the filter predicate.
        - selectivity (float): The estimated percentage of documents that match the filter (0.0 to 1.0).-
        - projected_keys (list of str, optional): the list of fields to retrieve in the output. If None or ["*"], the whole document is returned.
        - sharding_key (str, optional): the key used to distribute the data across servers.
                                        If filter_key == sharding_key, the cost is divided by nb_servers.
        - nb_servers (int): The number of servers in the cluster (default: 1000)
        """
        # 1. Output Size (Network)
        # We calculate the size of the result document (Projected)
        out_doc_size_bytes = collection.get_projected_doc_size(projected_keys)
        out_docs_count = collection.count * selectivity
        
        output_total_bytes = out_docs_count * out_doc_size_bytes

        # 2. Input Scanned Size (Disk I/O)
        # Default: Full Scan (We read everything)
        total_scan_bytes = collection.doc_size_bytes * collection.count
        
        # 3. Sharding Impact
        servers_hit = 1
        comment = ""
        
        if sharding_key is None:
            # Case: No Sharding (Centralized or Full Broadcast on replicas)
            servers_hit = 1 
            comment = "No Sharding (Full Scan)"
            
        else:
            if filter_key == sharding_key:
                # Case: Targeted Query
                # We only scan the relevant server (1/N of the data)
                total_scan_bytes = total_scan_bytes / nb_servers
                servers_hit = 1
                comment = f"Targeted Query on Shard Key '{sharding_key}'"
            else:
                # Case: Broadcast Query
                # We scan everything, but distributed across N servers.
                # Total volume scanned is still high, but latency is lower (parallelism).
                servers_hit = nb_servers
                comment = f"Broadcast Query (Filter '{filter_key}' != Shard '{sharding_key}')"

        return {
            "operator": "Filter",
            "comment": comment,
            "scanned_gb": total_scan_bytes / (1024**3), # Direct conversion
            "output_docs": int(out_docs_count),
            "output_gb": output_total_bytes / (1024**3),
            "servers_involved": servers_hit
        }
    
    @staticmethod
    def op_nested_loop(outer_col, inner_col, join_key, sharding_key_inner=None, nb_servers=1000):
        """
        Operator 3 and 4 - Nested loop join

        Args : 
        - outer_col (Collection object): the "Driving" collection. We iterate through every document of this collection.
        - inner_col (Collection object): The "Driven" (or Lookup) collection. We search inside this collection for every iteration.
        - join_key (str): The common attribute used to join the two collections (e.g., "idp" or "product_id").
        - sharding_key_inner (str, optional): The sharding key of the inner collection.
        - nb_servers (int)
        """
        # Number of lookups / loops. 
        outer_loops = outer_col.count
        
        # Cost of a single lookup in the inner
        # it depends on the sharding : 
        # If join_key = sharding_key --> we look only in 1 server
        # Else : broadcast
        cost_per_lookup = 0
        
        if sharding_key_inner and join_key == sharding_key_inner:
            # Targeted Lookup : We know where is the data
            # Cost = Size of an inner document or partial scan
            cost_per_lookup = (inner_col.doc_size_bytes * inner_col.count) / nb_servers
            comment = "Optimized Join (Key = Shard Key)"
        else:
            # Broadcast Lookup : for all lines of outer, we request all servers
            cost_per_lookup = inner_col.doc_size_bytes * inner_col.count
            comment = "Broadcast Join (Very Expensive)"

        total_scanned_bytes = (outer_col.doc_size_bytes * outer_col.count) + (outer_loops * cost_per_lookup)

        return {
            "operator": "Nested Loop Join",
            "comment": comment,
            "outer_collection": outer_col.name,
            "inner_collection": inner_col.name,
            "total_scanned_gb": total_scanned_bytes / (1024**3)
        }
    
    @staticmethod
    def op_aggregate(collection, group_key, group_cardinality, projected_keys=None, filter_key=None, selectivity=1.0, sharding_key=None, nb_servers=1000):
        """
        Operator 5: Aggregate/ Group By ( with/without sharding)
        
        Args:
        - collection: The targeted collection.
        - group_key: The grouping key.
        - group_cardinality: The number of distinct values for the group_key.
        - projected_keys: The expected output format.
        - filter_key: An optional filtered key applied before grouping.
        - selectivity: How many docs pass the filter.
        - sharding_key: The key used to distribute the data.
        """
        # 1. Filter & Scan Phase (Map)
        # We compute how many documents disk has to read to do the aggregation
        # It depends on the condition : filter_key == sharding_key or not
        input_docs = collection.count * selectivity
        
        # Scan cost logic
        if filter_key and filter_key == sharding_key:
            # Filter matches sharding: we scan only 1 server
            scanned_bytes = (collection.doc_size_bytes * collection.count) / nb_servers
            comment = f"Targeted Scan on '{filter_key}'"
        else:
            # Full scan across all servers
            scanned_bytes = collection.doc_size_bytes * collection.count
            comment = "Full Scan"

        # 2. Shuffle Phase (Network Transfer for Grouping)
        # If we group by the sharding key, the data is already in the right place!
        # Otherwise, we must send key-value pairs across the network to group them into 1 server.
        map_kv_size = 24  # Estimate size of a mapped pair / cost of the transit
        
        if group_key == sharding_key:
            shuffle_bytes = 0
            comment += " -> Local Reduce (No Shuffle)"
        else:
            shuffle_bytes = input_docs * map_kv_size
            comment += " -> Network Shuffle Required"

        # 3. REDUCE PHASE (Output)
        # The number of output documents is the number of distinct groups.
        # But if we filtered heavily, we can't have more groups than input documents!
        out_docs_count = min(group_cardinality, input_docs)
        
        out_doc_size_bytes = collection.get_projected_doc_size(projected_keys)
        output_total_bytes = out_docs_count * out_doc_size_bytes

        return {
            "operator": "Aggregate",
            "comment": comment,
            "scanned_gb": scanned_bytes / (1024**3),
            "shuffle_gb": shuffle_bytes / (1024**3), # Cost estimating data shuffle
            "output_docs": int(out_docs_count),      # Number of distinct key values
            "output_gb": output_total_bytes / (1024**3)
        }