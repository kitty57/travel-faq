import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set up the title and description of the app
st.title("🌍 TravelBot - Your Travel Assistant")
st.write("Ask me anything about travel destinations, visas, or the best time to visit!")

# Your Hugging Face API key
HF_API_KEY = "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"  # Replace with your actual API key

# Load the model and tokenizer from Hugging Face with API key
@st.cache_resource  # Cache the model and tokenizer for faster reloads
def load_model():
    model_name = "kitty528/travelbot"  # Replace with your model's name
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        use_auth_token=HF_API_KEY  # Use API key for authentication
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_auth_token=HF_API_KEY
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

model, tokenizer = load_model()

# Input box for user query
user_input = st.text_input("Ask your travel question here:")

# Generate response when the user submits a question
if user_input:
    # Prepare the input messages
    messages = [{"role": "user", "content": user_input}]

    # Tokenize the input
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    # Create attention mask
    attention_mask = inputs != tokenizer.pad_token_id

    # Generate a response
    with st.spinner("Generating response..."):  # Show a spinner while generating
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=128,  # Adjust based on your needs
            use_cache=True,
            temperature=0.6,
            min_p=0.1,
        )

    # Decode the generated tokens into human-readable text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display the response
    st.write("**TravelBot:**")
    st.write(response)
