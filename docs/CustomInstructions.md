# Navigate to your desired parent directory first, then run:

# Create root project directory
New-Item -ItemType Directory -Path "retail_demand_analysis"
Set-Location "retail_demand_analysis"

# Create main folders
New-Item -ItemType Directory -Path "notebooks"
New-Item -ItemType Directory -Path "data/raw"
New-Item -ItemType Directory -Path "data/processed"
New-Item -ItemType Directory -Path "data/results/eda"
New-Item -ItemType Directory -Path "data/results/features"
New-Item -ItemType Directory -Path "data/results/models"
New-Item -ItemType Directory -Path "outputs/figures/eda"
New-Item -ItemType Directory -Path "outputs/figures/features"
New-Item -ItemType Directory -Path "outputs/figures/models"
New-Item -ItemType Directory -Path "docs/plans"
New-Item -ItemType Directory -Path "docs/decisions"
New-Item -ItemType Directory -Path "docs/reports"
New-Item -ItemType Directory -Path "presentation"

# Create .gitkeep files to preserve empty directories in Git
New-Item -ItemType File -Path "data/raw/.gitkeep"
New-Item -ItemType File -Path "data/processed/.gitkeep"
New-Item -ItemType File -Path "data/results/eda/.gitkeep"
New-Item -ItemType File -Path "data/results/features/.gitkeep"
New-Item -ItemType File -Path "data/results/models/.gitkeep"
New-Item -ItemType File -Path "outputs/figures/eda/.gitkeep"
New-Item -ItemType File -Path "outputs/figures/features/.gitkeep"
New-Item -ItemType File -Path "outputs/figures/models/.gitkeep"

# Create .gitignore file
@"
# Data files
*.csv
*.pkl
*.parquet
*.7z
*.zip

# Jupyter Notebook checkpoints
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Keep structure files
!.gitkeep
"@ | Out-File -FilePath ".gitignore" -Encoding utf8

# Initialize Git repository
git init

# Verify structure
Get-ChildItem -Recurse -Directory | Select-Object FullName
```

**After running, you should see:**
```
retail_demand_analysis/
├── .git/
├── .gitignore
├── notebooks/
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│       ├── eda/
│       ├── features/
│       └── models/
├── outputs/
│   └── figures/
│       ├── eda/
│       ├── features/
│       └── models/
├── docs/
│   ├── plans/
│   ├── decisions/
│   └── reports/
└── presentation/