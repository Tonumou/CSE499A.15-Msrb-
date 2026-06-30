# Setup & Execution Instructions

### 1. Create a Virtual Environment
You can name your virtual environment anything you and your collaborators prefer. Replace `<your_env_name>` with your chosen name (e.g., `venv`, `Virtual`, `env`).
```powershell
python -m venv <your_env_name>
```

### 2. Activate the Virtual Environment
```powershell
.\<your_env_name>\Scripts\activate
```

### 3. Install Project Dependencies
```powershell
pip install -r requirements.txt
```
*(Note: Our dynamic pipeline scripts are self-bootstrapping and will automatically install missing dependencies, but this step ensures your entire workspace is ready).*

### 4. Set Your Gemini API Key Securely
We use strict system environment variables to prevent accidental key leaks. No `.env` files are allowed in this pipeline.
**If using PowerShell:**
```powershell
$env:GEMINI_API_KEY="your_actual_api_key_here"
```
**If using Command Prompt (CMD):**
```cmd
set GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Standardize AIDER Images
This step processes the raw AIDER images, standardizes their resolution/colors, drops grayscale outliers, and prevents Out-Of-Memory errors on GPUs.
```powershell
python support/aider_image_standardizer.py
```

### 6. Synthesize Labels for AIDER
This step uses the Gemini Vision API via the `google.genai` SDK to analyze the standardized images and output rich, xBD-style JSON metadata.
```powershell
python support/aider_synthesize_metadata.py
```

### 7. Generate Final Ground Truth Dataset
Finally, run the dedicated AIDER generation script to aggregate all processed images and synthesized labels into a finalized multimodal dataset.

```powershell
python support/aider_generate_ground_truth.py
```