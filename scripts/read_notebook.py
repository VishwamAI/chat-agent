import nbformat

def read_notebook(notebook_path):
    with open(notebook_path, 'r') as f:
        notebook = nbformat.read(f, as_version=4)
    return notebook

def print_notebook_contents(notebook):
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            print("Code Cell:")
            print(cell.source)
        elif cell.cell_type == 'markdown':
            print("Markdown Cell:")
            print(cell.source)
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    notebook_path = "../mathematics_conjectures/knot_theory.ipynb"
    notebook = read_notebook(notebook_path)
    print_notebook_contents(notebook)
