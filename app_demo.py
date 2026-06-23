"""
GridLock — Entry Point (app_demo.py)
=====================================
Navigation entry point using st.navigation() (Streamlit 1.36+).
All page content lives in demo_content.py and pages/1_ML_Models.py.

Run:  streamlit run app_demo.py
"""
import streamlit as st

st.set_page_config(
    page_title="GridLock | Team Srikrit",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

demo = st.Page("demo_content.py", title="GridLock Demo", icon="🚦", default=True)
ml   = st.Page("pages/1_ML_Models.py", title="ML Architecture", icon="🧠")

pg = st.navigation([demo, ml])
pg.run()
