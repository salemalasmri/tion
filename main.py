import streamlit as st
import requests
import traceback

st.set_page_config(page_title="EzOptions", layout="wide")

url = "https://raw.githubusercontent.com/EazyDuz1t/EzOptions/main/ezoptions.py"

try:
    response = requests.get(url)
    code = response.text
    exec(code, globals())

except Exception:
    st.error("Runtime Error:")
    st.code(traceback.format_exc())
