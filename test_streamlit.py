import streamlit as st

st.title("ResearchLens AI")
st.success("Streamlit is working!")

name = st.text_input("Enter your name")

if st.button("Submit"):
    st.write("Hello,", name)
