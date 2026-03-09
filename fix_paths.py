import os
import glob

# Fix paper.tex
paper_path = r"d:\ALL MY ML AND AI\paper\paper.tex"
if os.path.exists(paper_path):
    with open(paper_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Update image paths
    images = ["fig1_training_loss.png", "fig2_gravity_invariance.png", "fig3_sindy_equations.png", 
              "fig5_sindy_rollout.png", "fig_exp4_architecture_comparison.png", "fig_exp6_triple_composition.png",
              "paper_money_plot.png", "spring_trajectory.png", "multi_step_rollout.png", "combined_trajectory.png"]
    
    for img in images:
        content = content.replace("{" + img + "}", "{figures/" + img + "}")
        
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(content)

# Fix python scripts loaded models
py_files = glob.glob(r"d:\ALL MY ML AND AI\src\*.py")
for py_file in py_files:
    with open(py_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple replacement assuming paths are loaded directly
    content = content.replace("'brain1_", "'../models/brain1_").replace('"brain1_', '"../models/brain1_')
    content = content.replace("'brain2_", "'../models/brain2_").replace('"brain2_', '"../models/brain2_')
    
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(content)

print("Paths fixed successfully.")
