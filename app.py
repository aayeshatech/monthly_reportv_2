import streamlit as st
from datetime import date

def main():
    st.set_page_config(
        page_title="Astro Trading Report",
        layout="wide"
    )
    st.title("✨ Astrological Trading Report")
    st.write(f"Today is {date.today().strftime('%B %d, %Y')}")
    
    if st.button("Test Connection"):
        st.success("Successfully connected!")

if __name__ == "__main__":
    main()
