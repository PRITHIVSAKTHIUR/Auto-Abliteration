import os
import random
import torch
from tqdm import tqdm
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- Helper functions ---

def load_instructions(dataset_id, column, n_instructions):
    dataset = load_dataset(dataset_id, split="train")
    indices = random.sample(range(len(dataset)), n_instructions * 2)
    return [dataset[i][column] for i in indices[:n_instructions]], [
        dataset[i][column] for i in indices[n_instructions:]
    ]

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        min_p=0.1,
        repetition_penalty=1.05,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def generate_outputs(model, tokenizer, instructions, system_prompt):
    outputs = []
    for instruction in tqdm(instructions, desc="Generating outputs", leave=False):
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            prompt = system_prompt + "\n" + instruction
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            inputs,
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        outputs.append(out["hidden_states"][0])
    return outputs

def orthogonalize_matrix(matrix, vec, weight):
    vec = vec.view(-1).to(matrix.device)
    if matrix.shape[-1] == vec.shape[0]:
        proj = torch.einsum("...d,d->...", matrix, vec).unsqueeze(-1) * vec.unsqueeze(0)
        return matrix - weight * proj
    elif matrix.shape[0] == vec.shape[0]:
        proj = torch.einsum("d...,d->...", matrix, vec).unsqueeze(0) * vec.unsqueeze(-1)
        return matrix - weight * proj
    else:
        raise ValueError(
            f"Matrix shape {matrix.shape} incompatible with vector shape {vec.shape}"
        )

# --- Streamlit UI ---

st.title("LLM Auto Abliteration")
st.markdown("🥠 Recommended for edge-device LLMs (e.g., 1B, 1.5B, 0.5B).")
st.markdown("🥠 Duplicate the space for seamless usage!")
st.markdown("🥠 This app allows you to manually input parameters to modify a language model's behavior by abliterating its weights.")
st.markdown("📍 Credits: Thanks to **[Maxime Labonne](https://huggingface.co/mlabonne)**")

# Debugging window to show log messages
debug_log = []
debug_placeholder = st.empty()
def update_debug(msg):
    debug_log.append(msg)
    debug_placeholder.text("\n".join(debug_log))

# Sidebar parameters
st.sidebar.header("Abliteration Parameters")
MODEL_ID = st.sidebar.text_input("Model ID", "prithivMLmods/FastThink-0.5B-Tiny")
N_INSTRUCTIONS = st.sidebar.number_input("Number of Instructions", min_value=1, value=128, step=1)
TARGET_LAYER = st.sidebar.slider("Target Layer (relative ratio)", 0.0, 1.0, 0.65, step=0.05)
REFUSAL_WEIGHT = st.sidebar.slider("Refusal Weight", 0.0, 2.0, 1.0, step=0.05)
PRIVATE_UPLOAD = st.sidebar.checkbox("Push Model to Hub Privately", value=True)

st.sidebar.header("HF Token")
hf_token = st.sidebar.text_input("Hugging Face Token", type="password")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    update_debug("HF Token received.")

st.sidebar.header("Target Dataset")
target_prompt = st.sidebar.text_area("Target Prompt", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
target_dataset = st.sidebar.text_input("Target Dataset ID", "mlabonne/harmful_behaviors")
target_column = st.sidebar.text_input("Target Column Name", "text")

st.sidebar.header("Baseline Dataset")
baseline_prompt = st.sidebar.text_area("Baseline Prompt", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")
baseline_dataset = st.sidebar.text_input("Baseline Dataset ID", "mlabonne/harmless_alpaca")
baseline_column = st.sidebar.text_input("Baseline Column Name", "text")

if st.button("Run Abliteration"):
    update_debug("Starting abliteration process...")
    
    st.write("### Loading Model and Tokenizer")
    update_debug("Checking device and GPU properties.")
    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
            update_debug("Using bfloat16 and flash_attention_2.")
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"
            update_debug("Using float16 and eager attention.")
    else:
        torch_dtype = torch.float32
        attn_implementation = "eager"
        update_debug("No GPU available. Using float32.")
    
    with st.spinner("Loading model..."):
        try:
            # Pass HF token if provided to load private or restricted models
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
                use_auth_token=hf_token if hf_token else None
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                use_auth_token=hf_token if hf_token else None
            )
            update_debug("Model and tokenizer loaded successfully.")
        except Exception as e:
            update_debug(f"Error loading model: {e}")
            st.error("Failed to load model. Check the debug log for details.")
            st.stop()
    
    try:
        total_layers = len(model.model.layers)
        update_debug(f"Model has {total_layers} layers.")
    except AttributeError:
        st.error("The loaded model does not have a `model.layers` attribute.")
        update_debug("Model does not have `model.layers` attribute. Aborting.")
        st.stop()
    LAYER_IDX = int(TARGET_LAYER * total_layers)
    update_debug(f"Selected layer index: {LAYER_IDX} based on relative ratio {TARGET_LAYER}.")
    
    st.write("### Before Abliteration Response")
    prompt_example = "How to write a computer virus?"
    response_before = generate_response(model, tokenizer, prompt_example)
    st.text_area("Response before abliteration:", response_before, height=150)
    update_debug("Generated response before abliteration.")
    
    st.write("### Loading Instructions")
    with st.spinner("Loading target instructions..."):
        try:
            target_instructions, _ = load_instructions(target_dataset, target_column, N_INSTRUCTIONS)
            update_debug("Target instructions loaded.")
        except Exception as e:
            update_debug(f"Error loading target instructions: {e}")
            st.error("Failed to load target instructions.")
            st.stop()
    with st.spinner("Loading baseline instructions..."):
        try:
            baseline_instructions, _ = load_instructions(baseline_dataset, baseline_column, N_INSTRUCTIONS)
            update_debug("Baseline instructions loaded.")
        except Exception as e:
            update_debug(f"Error loading baseline instructions: {e}")
            st.error("Failed to load baseline instructions.")
            st.stop()
    
    st.write("### Generating Hidden States")
    with st.spinner("⌛ Generating the baseline hidden state. Hold tight, as this may take 10 minutes or more."):
        baseline_outputs = generate_outputs(model, tokenizer, baseline_instructions, system_prompt=baseline_prompt)
        update_debug("Baseline hidden states generated.")
    with st.spinner("Generating target hidden states..."):
        target_outputs = generate_outputs(model, tokenizer, target_instructions, system_prompt=target_prompt)
        update_debug("Target hidden states generated.")
    
    target_hidden = [output[LAYER_IDX][:, -1, :] for output in target_outputs]
    baseline_hidden = [output[LAYER_IDX][:, -1, :] for output in baseline_outputs]
    update_debug("Extracted last token hidden states.")
    
    st.write("### Calculating Refusal Direction")
    target_mean = torch.stack(target_hidden).mean(dim=0)
    baseline_mean = torch.stack(baseline_hidden).mean(dim=0)
    refusal_dir = target_mean - baseline_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    update_debug("Calculated and normalized the refusal direction.")
    
    del target_outputs, baseline_outputs, target_hidden, baseline_hidden
    
    st.write("### Orthogonalizing Model Weights")
    refusal_dir = refusal_dir.view(-1).to(model.device)
    stats = {"embed_tokens": False, "attention_o_proj": 0, "mlp_proj": 0}
    
    if hasattr(model.model, "embed_tokens"):
        model.model.embed_tokens.weight.data = orthogonalize_matrix(
            model.model.embed_tokens.weight.data, refusal_dir, REFUSAL_WEIGHT
        )
        stats["embed_tokens"] = True
        update_debug("Orthogonalized embed_tokens weights.")
    
    for layer in tqdm(model.model.layers, desc="Orthogonalizing weights", leave=False):
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            layer.self_attn.o_proj.weight.data = orthogonalize_matrix(
                layer.self_attn.o_proj.weight.data, refusal_dir, REFUSAL_WEIGHT
            )
            stats["attention_o_proj"] += 1
        if hasattr(layer, "mlp"):
            proj_name = (
                "down_proj"
                if hasattr(layer.mlp, "down_proj")
                else "c_proj"
                if hasattr(layer.mlp, "c_proj")
                else None
            )
            if proj_name:
                getattr(layer.mlp, proj_name).weight.data = orthogonalize_matrix(
                    getattr(layer.mlp, proj_name).weight.data, refusal_dir, REFUSAL_WEIGHT
                )
                stats["mlp_proj"] += 1
    update_debug("Orthogonalized layer weights.")
    
    del refusal_dir
    
    if (
        not stats["embed_tokens"]
        and stats["attention_o_proj"] == 0
        and stats["mlp_proj"] == 0
    ):
        st.error("Failed to orthogonalize any model weights. Model not abliterated.")
        update_debug("No weights were orthogonalized. Aborting process.")
        st.stop()
    
    update_debug(f"Orthogonalization stats: {stats}")
    st.write(f"Orthogonalization stats: {stats}")
    
    st.write("### After Abliteration Response")
    response_after = generate_response(model, tokenizer, prompt_example)
    st.text_area("Response after abliteration:", response_after, height=150)
    update_debug("Generated response after abliteration.")
    
    st.write("### Pushing Model to Hugging Face Hub")
    try:
        model_name = MODEL_ID.split("/")[-1] + "-abliterated"
        model.push_to_hub(model_name, private=PRIVATE_UPLOAD)
        tokenizer.push_to_hub(model_name, private=PRIVATE_UPLOAD)
        st.success(f"Model automatically pushed as {model_name}")
        update_debug(f"Model automatically pushed to HF Hub as {model_name}.")
    except Exception as e:
        st.error(f"Error while pushing model: {e}")
        update_debug(f"Error while pushing model: {e}")
    
    st.success("Abliteration process complete!")
    update_debug("Abliteration process complete.")
