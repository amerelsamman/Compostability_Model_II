import streamlit as st
import pandas as pd
import numpy as np

# Simple test app
st.title("🧪 Streamlit Deployment Test")
st.write("This is a minimal test to check if Streamlit Cloud is working.")

# Test basic functionality
st.subheader("Basic Functionality Test")
st.write("✅ Streamlit is running")
st.write("✅ Pandas is working")

# Test data creation
df = pd.DataFrame({
    'x': np.arange(10),
    'y': np.random.randn(10)
})
st.write("✅ NumPy is working")
st.dataframe(df)

# Test plotting
st.subheader("Plotting Test")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['A', 'B', 'C']
)
st.line_chart(chart_data)

st.success("🎉 All tests passed! Streamlit Cloud is working.")
st.info("If you can see this, the deployment is successful!") 