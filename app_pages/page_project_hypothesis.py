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
        f"* The ML model successfully learned to distinguish these features with high accuracy (>97%), validating that the visual difference is sufficient for classification."
    )

    st.write("---")
    st.write("### Empirical Evidence Analysis")
    
    if st.checkbox("Show Average and Variability Study"):
        st.image("out/visualization/avg_var_healthy.png", caption='Healthy Leaf - Average and Variability')
        st.image("out/visualization/avg_var_powdery_mildew.png", caption='Powdery Mildew Leaf - Average and Variability')
        
        st.warning(
            f"The visual analysis confirms that infected leaves display whitish powdery patches "
            f"and higher texture variability, which are key indicators the model uses for classification."
        )

    if st.checkbox("Show Dataset Class Balance"):
        st.write("### Dataset Class Balance")
        st.info(
            f"A balanced dataset is crucial for training a reliable model. "
            f"It ensures the model learns to identify both classes with equal precision."
        )
        # Class counts verified from data directory
        data = {"Healthy": 2104, "Powdery Mildew": 2104}
        st.bar_chart(data)
        st.write(f"The dataset contains **2,104** images of healthy leaves and **2,104** images of infected leaves, representing a perfect 50/50 balance.")
