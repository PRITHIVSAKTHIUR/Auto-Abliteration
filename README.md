# **Auto Abliteration**

https://github.com/user-attachments/assets/cd182313-dcb4-4d72-aee7-c324aca10091

Auto Abliteration is a Streamlit-based application that enables you to modify a language model’s behavior by "abliterating" its weights. This tool is especially recommended for edge-device LLMs (e.g., 0.5B, 1B, 1.5B models). By orthogonalizing key model weights along a computed refusal direction, Auto Abliteration can subtly alter the model’s responses.

## Features

- **Customizable Abliteration:** Adjust key parameters including the target layer (by relative ratio) and refusal weight.
- **Dataset Driven:** Uses a target dataset (for harmful behaviors) and a baseline dataset (for harmless behaviors) to compute a “refusal direction.”
- **Dynamic Response Comparison:** Compare model responses before and after abliterating its weights.
- **Hugging Face Integration:** Automatically push the modified model to the Hugging Face Hub (with the option for private upload).
- **Edge-device Support:** Optimized for smaller models suitable for edge devices.

## Architecture Overview

The application uses several helper functions:

- **`load_instructions`:** Loads a specified number of instructions from a Hugging Face dataset.
- **`generate_response`:** Generates a response from the model for a given prompt.
- **`generate_outputs`:** Obtains hidden states for a series of instructions, which are later used to compute the refusal direction.
- **`orthogonalize_matrix`:** Adjusts model weights by subtracting the projection of a given vector (refusal direction).

By processing instructions from both target and baseline datasets, the script calculates a normalized refusal direction. Then, it orthogonalizes the weights (e.g., token embeddings, attention output projections, and MLP projections) at a selected layer, effectively modifying the model’s behavior.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/Auto-Abliteration.git
   cd Auto-Abliteration
   ```

2. **Set up a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Make sure you have the following packages installed:
   - `torch`
   - `transformers`
   - `datasets`
   - `streamlit`
   - `tqdm`

## Usage

1. **Configure Abliteration Parameters:**

   When you run the app, use the sidebar to:
   - Enter the model ID (e.g., `prithivMLmods/FastThink-0.5B-Tiny`).
   - Specify the number of instructions to use.
   - Adjust the target layer (as a relative ratio) and refusal weight.
   - Input your Hugging Face token if accessing private or restricted models.
   - Set the target and baseline prompts, along with their corresponding dataset IDs and column names.

2. **Run the Streamlit App:**

   Launch the app using:

   ```bash
   streamlit run <your_script_name>.py
   ```

   Replace `<your_script_name>.py` with the filename containing your code.

3. **Workflow:**

   - The app first loads the model and tokenizer.
   - It generates an initial response for a sample prompt (e.g., "How to write a computer virus?").
   - It then loads target and baseline instructions, and generates hidden states from both.
   - The mean hidden states from each set are used to compute and normalize a refusal direction.
   - Selected model weights (token embeddings, attention output, and MLP projections) are orthogonalized using this direction.
   - A new response is generated to showcase the change.
   - Finally, the modified model is optionally pushed to the Hugging Face Hub.

## Example

When you run the application, you might see two sections:

- **Before Abliteration Response:** Shows the model’s original response.
- **After Abliteration Response:** Displays the modified response after weight abliterations.

Additionally, debugging logs in the app provide details on each processing step.

## Credits

- Thanks to [Maxime Labonne](https://huggingface.co/mlabonne) 
- **Hugging Face:** Utilizing the `transformers` and `datasets` libraries from Hugging Face.
