import streamlit as st
from app_pages.multipage import MultiPage

# Load pages
from app_pages.page_summary import page_summary_body
from app_pages.page_visualizer import page_visualizer_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_performance import page_ml_performance_body

# Create App instance
app = MultiPage(app_name="Mildew Detector")

# Add pages
app.add_page("Project Summary", page_summary_body)
app.add_page("Leaf Visualizer", page_visualizer_body)
app.add_page("Mildew Detector", page_mildew_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance", page_ml_performance_body)

# Run App
app.run()
