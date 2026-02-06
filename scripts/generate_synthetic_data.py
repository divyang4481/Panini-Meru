import json
import random
import os


def generate_deep_nesting(depth=10, content_lines=50):
    """
    Generates code that is deeply nested to tire out the attention mechanism.
    """
    code = []
    # Indent
    for i in range(depth):
        indent = "    " * i
        code.append(f"{indent}if condition_{i}:")

    # Body
    final_indent = "    " * depth
    for j in range(content_lines):
        code.append(f"{final_indent}x_{j} = compute_stuff({j})")
        code.append(f"{final_indent}# Filler line to extend context length...")

    # Dedent (Adelic Memory should track this!)
    # We explicitly write the deducation to teach the model to 'pop' the stack.
    for i in range(depth - 1, -1, -1):
        # We might not explicitly write code here in python (whitespace matters),
        # but for training data, having subsequent code at lower levels helps.
        indent = "    " * i
        code.append(f"{indent}post_process_level_{i}()")

    return "\n".join(code)


def generate_long_range_ref(distractors=500):
    """
    Defines a variable, fills standard context window with noise, then uses it.
    """
    magic_val = random.randint(1000, 9999)
    var_name = f"config_param_{random.randint(0, 100)}"

    code = [f"{var_name} = {magic_val}"]

    # Noise
    for i in range(distractors):
        code.append(f"def filler_func_{i}():")
        code.append(f"    pass # Just filling space: {random.random()}")
        code.append("")

    # Usage
    code.append(f"def critical_logic():")
    code.append(f"    # The model must recall {var_name}")
    code.append(f"    return {var_name}")

    return "\n".join(code)


def main():
    output_file = "data/synthetic_hard_cases.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Generating synthetic 'Hard Cases' to {output_file}...")

    with open(output_file, "w") as f:
        # Generate 100 Deep Nesting examples
        for _ in range(100):
            entry = {
                "text": generate_deep_nesting(
                    depth=random.randint(8, 20), content_lines=100
                ),
                "meta": "deep_nesting",
            }
            f.write(json.dumps(entry) + "\n")

        # Generate 100 Long Range examples
        for _ in range(100):
            entry = {
                "text": generate_long_range_ref(distractors=200),  # ~1000 tokens
                "meta": "long_range_memory",
            }
            f.write(json.dumps(entry) + "\n")

    print("Done. You can mix this dataset into your training loop.")


if __name__ == "__main__":
    main()
