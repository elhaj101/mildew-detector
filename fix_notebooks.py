import nbformat as nbf
import os

def add_headers_to_notebook(filepath, data_collection=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
    
    # Add main header to the first cell if it doesn't have it
    if nb.cells:
        content = nb.cells[0].source
        if "Objectives" not in content:
            nb.cells[0].source = f"# {content.split('\n')[0]}\n\n## Objectives\n- Standardize project documentation\n\n## Inputs\n- Raw data\n\n## Outputs\n- Processed data\n" + "\n".join(content.split('\n')[1:])

    # Inject endpoint data collection if it's DataCollection.ipynb
    if data_collection:
        new_cells = []
        for cell in nb.cells:
            new_cells.append(cell)
            if cell.cell_type == 'markdown' and '## Define dataset paths' in cell.source:
                code = "import urllib.request\nimport zipfile\n\n# Endpoint for data collection\ndata_url = 'https://github.com/elhaj101/mildew-detector/archive/refs/heads/main.zip'\ndestination = '../data/raw_data.zip'\n\n# Download if not exists\nif not os.path.exists(destination):\n    print('Downloading data...')\n    urllib.request.urlretrieve(data_url, destination)\n    with zipfile.ZipFile(destination, 'r') as zip_ref:\n        zip_ref.extractall('../data/')\n"
                new_cells.append(nbf.v4.new_code_cell(code))
        nb.cells = new_cells

    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

notebooks = [
    ('jupyter_notebooks/DataCollection.ipynb', True),
    ('jupyter_notebooks/Visualization.ipynb', False),
    ('jupyter_notebooks/ModelingandEvaluation.ipynb', False)
]

for path, is_dc in notebooks:
    if os.path.exists(path):
        print(f"Fixing {path}...")
        add_headers_to_notebook(path, is_dc)
