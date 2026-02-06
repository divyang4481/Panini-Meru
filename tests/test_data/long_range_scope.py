
def long_range_scope_test():
    """
    Test Case: Long-Range Variable Adherence
    
    The variable `user_request_id` is defined at the very top.
    We then have 150 lines of distracting comments and unrelated code.
    Finally, we ask the model to print the ID.
    
    Failure Mode (Base Model): Hallucinates a new variable name (e.g. `request_id`, `id`, `uid`) 
    because the definition is too far back or lost in attention noise.
    
    Success Mode (PMeru): The Prime Stream maintains the symbol table state and correctly predicts `user_request_id`.
    """
    
    # 1. Definition
    user_request_id = "REQ-12345"
    
    # ... [Insert 200 lines of noise] ...
    # This simulates a long function body or a large file.
    # The model must hold "user_request_id" in working memory.
    
    pass
    pass
    # ...
    
    # 2. Retrieval
    print(  # Model should complete: user_request_id)
