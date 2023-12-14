import streamlit as st
import numpy as np
from transformers import GPT2LMHeadModel, AutoTokenizer, pipeline


config = {
    "max_length": 250,
    "temperature": 1.1,
    "top_p": 2.,
    "num_beams": 10,
    "repetition_penalty": 1.5,
    "num_return_sequences": 9,
    "no_repeat_ngram_size": 2,
    "do_sample": True
}


model_name = 'ai-forever/rugpt3small_based_on_gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name).to('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)


generation = pipeline('text-generation', model=model,
                      tokenizer=tokenizer, device=-1)


def generate_report(input):
    generated_output = generation(input, **config)[0]['generated_text']
    return generated_output

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Генерация аннотации и введения для КНИР')

input_text = st.text_area('Введите стартовую фразу:', '', height=200, key='example_text')

start_generation = st.button("Сгенерировать")

if start_generation: 
    with st.spinner("Генерация..."):
        generated_response = generate_report(input_text)
        st.write(generated_response)
