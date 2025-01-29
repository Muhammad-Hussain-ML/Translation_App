import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load the model and tokenizer
MODEL_NAME = "abdulwaheed1/english-to-urdu-translation-mbart"
@st.cache_resource
def load_model():
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("English to Urdu Translator")
st.write("Enter an English sentence, and the app will translate it into Urdu.")

# User input
text = st.text_area("Enter text in English:", "")

if st.button("Translate"):
    if text.strip():
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # Generate translation
        output_ids = model.generate(**inputs)
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Display translation
        st.success("Translation:")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
