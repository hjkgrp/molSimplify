# This script alphabetizes the entries in ligands.dict
# Assumes the first two lines are comments.

def main():
    with open('ligands.dict', 'r') as f:
        array_content = f.readlines()

    comment_lines = array_content[:2]
    dictionary_lines = array_content[2:]
    dictionary_lines.sort()  # Alphabetization

    # Rewriting ligands.dict, alphabetized.
    with open('ligands.dict', 'w') as f:
        for line in comment_lines:
            f.write(line)
        for line in dictionary_lines:
            f.write(line)


if __name__ == "__main__":
    main()
