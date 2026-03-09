import json
import os

notebook_path = os.path.join("d:\\ALL MY ML AND AI", "CASM_Brain1_Full.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        changed = False
        for line in cell['source']:
            if "sim.state[0] = random.uniform" in line:
                new_source.append(line.replace("sim.state[0]", "sim.y"))
                changed = True
            elif "sim.state[1] = random.uniform" in line:
                new_source.append(line.replace("sim.state[1]", "sim.v"))
                changed = True
            elif "y_t, v_t = sim.state.copy()" in line:
                new_source.append(line.replace("sim.state.copy()", "sim.y, sim.v"))
                changed = True
            elif "y_t1, v_t1 = sim.state.copy()" in line:
                new_source.append(line.replace("sim.state.copy()", "sim.y, sim.v"))
                changed = True
            else:
                new_source.append(line)
        if changed:
            cell['source'] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook cell patched successfully.")
