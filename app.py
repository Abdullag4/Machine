import streamlit as st
import train
import predict

st.set_page_config(page_title="Shape Classifier", layout="centered")
st.title("🔵🟥 Shape Classifier")

tab1, tab2 = st.tabs(["📚 Train Model", "🔍 Test Model"])

with tab1:
    train.train_tab()

with tab2:
    predict.predict_tab()
