import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = 'ai4bharat/indictrans2-en-ur'

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        translated_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # Display translation
        st.success("Translation:")
        st.write(translated_text)
    else:
        st.warning("Please enter some text to translate.")
