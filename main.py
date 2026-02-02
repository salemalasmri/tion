import streamlit as st
import requests

st.set_page_config(page_title="EzOptions", layout="wide")

url = "https://raw.githubusercontent.com/EazyDuz1t/EzOptions/main/ezoptions.py"

response = requests.get(url)
code = response.text

exec(code)
