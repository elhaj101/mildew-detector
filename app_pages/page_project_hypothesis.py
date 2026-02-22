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

    st.write("---")
    st.write("### Empirical Evidence Analysis")
    
    if st.checkbox("Show Visual Analysis (Average vs. Difference)"):
        st.image("out/visualization/avg_var_healthy.png", caption='Healthy Leaf - Average and Variability')
        st.image("out/visualization/avg_var_powdery_mildew.png", caption='Powdery Mildew Leaf - Average and Variability')
        st.image("out/visualization/difference_avg.png", caption='Difference Image: Healthy vs. Infected')
        
        st.warning(
            f"The visual analysis confirms that infected leaves display whitish powdery patches "
            f"and higher texture variability, which are the primary indicators the model uses for classification."
        )
