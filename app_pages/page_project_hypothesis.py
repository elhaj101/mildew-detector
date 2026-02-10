import streamlit as st

def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.info(
        f"**Hypothesis**\n"
        f"* Infected leaves have distinct white powdery patches that differentiate them from healthy leaves."
    )

    st.success(
        f"**Validation**\n"
        f"* The Average Image study showed that infected leaves have lighter, whitish patterns compared to healthy leaves.\n"
        f"* The Difference Image analysis highlighted these patterns as the main differentiator.\n"
        f"* The ML model successfully learned to distinguish these features with high accuracy (>97%), validating that the visual difference is sufficient for classification."
    )
