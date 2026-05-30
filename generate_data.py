import pandas as pd

# Creating an expanded, high-variance dataset across multiple departments
data = {
    "complaint_text": [
        # Water Department
        "The main water pipeline burst on 5th avenue and water is flooding the street.",
        "There is zero water pressure in our building and a major leak in the main line.",
        "A broken water pipe is flooding the basement of our residential complex.",
        "Dirty water is coming out of our kitchen taps, please check the main filtration line.",
        
        # Sanitation Department
        "Garbage piles are rotting on the main road and the street smells terrible.",
        "The sanitation truck has missed our block's trash collection for two weeks.",
        "Overflowing public trash bins are attracting stray animals on the sidewalk.",
        "Illegal dumping of domestic waste in the neighborhood park needs cleanup.",
        
        # Electrical Department
        "A street light is flickering non-stop and the corner cul-de-sac is pitch black.",
        "A power line snapped and sparks are flying near the community center gate.",
        "The local transformer is making a loud buzzing sound and smoking slightly.",
        "An exposed electrical circuit breaker box on the sidewalk is wide open.",
        
        # Fire & Emergency/Safety Department (New Explicit Cluster)
        "An abandoned vehicle caught fire on the highway shoulder and thick smoke is rising.",
        "A car engine caught fire at the intersection, we need emergency crews immediately.",
        "A vehicle crashed into a pole and catches fire, send assistance right away.",
        "There is smoke coming out of an empty truck parked near the shopping complex."
    ],
    "assigned_officer": [
        "Officer_Water_Dept", "Officer_Water_Dept", "Officer_Water_Dept", "Officer_Water_Dept",
        "Officer_Sanitation_Dept", "Officer_Sanitation_Dept", "Officer_Sanitation_Dept", "Officer_Sanitation_Dept",
        "Officer_Electrical_Dept", "Officer_Electrical_Dept", "Officer_Electrical_Dept", "Officer_Electrical_Dept",
        "Officer_Emergency_Dept", "Officer_Emergency_Dept", "Officer_Emergency_Dept", "Officer_Emergency_Dept"
    ],
    "priority": [
        "High", "Medium", "High", "Medium",
        "Low", "Medium", "Low", "Low",
        "Low", "High", "High", "High",
        "High", "High", "High", "High"
    ],
    "eta": [
        2.0, 3.5, 1.5, 4.0,
        5.0, 3.0, 6.0, 5.5,
        5.0, 1.0, 1.5, 2.0,
        0.5, 0.5, 0.5, 1.0
    ]
}

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)
print("✅ Expanded, high-fidelity dataset generated successfully inside dataset.csv!")