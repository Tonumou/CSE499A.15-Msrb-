```markdown
# Setup & Execution Instructions

### 1. Create a Virtual Environment
```powershell
python -m venv Virtual
```

### 2. Activate the Virtual Environment
```powershell
.\Virtual\Scripts\activate
```

### 3. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Set Your Gemini API Keys
We now use a `.env` file to support multiple API keys for automatic rate-limit rotation.
Create a file named `.env` in the root of your project and add your valid `AIza...` keys:
```env
GEMINI_API_KEY_1="AIzaSyB..."
GEMINI_API_KEY_2="AIzaSyC..."
# Add up to 7 keys!
```

### 5. Standardize AIDER Images
This step processes the raw AIDER images, standardizes their resolution/colors, and prevents Out-Of-Memory errors on GPUs.
```powershell
python support/aider_image_standardizer.py
```

### 6. Synthesize Labels for AIDER
This step uses the Gemini Vision API to analyze the standardized images and output rich, xBD-style JSON metadata.
```powershell
python support/aider_label_synthesizer.py
```

### 7. Generate Final Ground Truth Dataset
Finally, run the original script to aggregate all processed images and synthesized labels into a finalized multimodal dataset.
```powershell
python support/generate_ground_truth.py
```
```