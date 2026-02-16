import json
import os

nb_path = 'jupyter_notebooks/Visualization.ipynb'

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    }

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        ntbk = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit(1)

# Check for existing cells
source_text = ""
for c in ntbk.get('cells', []):
    source_text += "".join(c.get('source', []))

if "Hypothesis Validation" in source_text:
    print("Hypothesis Validation cell already exists.")
    exit(0)

text = """## Hypothesis Validation

**Hypothesis:** Infected leaves have distinct white powdery patches that differentiate them from healthy leaves.

**Validation from Visualizations:**
*   **Average Image Analysis:** The average infected leaf image shows lighter/whitish potential patterns compared to the healthy average, though averaging spreads these out.
*   **Difference Analysis:** The difference between the average healthy and average infected leaf highlights the specific regions where the disease manifests. The high values in the difference plot correspond to the powdery mildew patches.
*   **Variability Analysis:** The variability images show where the features differ most among images of the same class.

Based on these visualizations, we observe clear differences in pixel intensity and distribution between the two classes, supporting the hypothesis that an ML model can learn to distinguish them.
"""

ntbk['cells'].append(create_markdown_cell(text))

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(ntbk, f, indent=1)

print(f"Appended Hypothesis Validation cell to {nb_path}")
