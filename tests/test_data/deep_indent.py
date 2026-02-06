
def nested_indentation_test():
    """
    Test Case: Deep Indentation Consistency
    
    Python relies on strict indentation.
    We create a heavily nested structure (5+ levels deep).
    
    Failure Mode (Base Model): Often forgets the current depth level after a long block, 
    un-indenting too early or changing indentation style (2 spaces vs 4 spaces).
    
    Success Mode (PMeru): Prime Stream explicitly tracks "depth=5", forcing the tokenizer to generate 
    the correct number of whitespace tokens before the next statement.
    """
    
    if True:
        if True:
            for i in range(10):
                try:
                    with open("file") as f:
                        if f.read():
                            # We are here (Depth 6)
                            # ... [Insert 100 lines of code at this depth] ...
                            
                            pass
                            pass
                            
                            # Model must strictly output indentation for Depth 6 here
                            return True
