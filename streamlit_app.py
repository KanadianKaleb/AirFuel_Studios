"""
streamlit_app.py

Main entry point for the Vehicle Log Tuner Streamlit application.
This file serves as the canonical entry point for Streamlit Cloud deployment.

To run the application:
    streamlit run streamlit_app.py

The application code is implemented in tuner_dashboard_Version21.py.
This wrapper provides a clean entry point for deployment with debug error handling.
"""

import streamlit as st
import traceback
import sys

# Safe debug entrypoint: Catch and display import/runtime errors in the browser
try:
    # Execute the main dashboard by importing it
    # Note: In Streamlit, importing a module executes it
    import tuner_dashboard_Version21
except Exception as e:
    # Display error information in the Streamlit UI instead of showing a blank page
    st.error("## Application Error")
    st.error(f"**Error Type:** {type(e).__name__}")
    st.error(f"**Error Message:** {str(e)}")
    
    # Display full traceback for debugging
    st.subheader("Full Traceback:")
    tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    st.code(tb_str, language="python")
    
    # Display Python and system information
    st.subheader("System Information:")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Python Executable:** {sys.executable}")
    
    # Display installed packages for debugging dependency issues
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            st.subheader("Installed Packages:")
            st.code(result.stdout, language="text")
    except Exception as pkg_error:
        st.write(f"Could not retrieve package list: {pkg_error}")
    
    # Provide helpful debugging tips
    st.subheader("Debugging Tips:")
    st.write("""
    1. Check if all required packages are installed: `pip install -r requirements.txt`
    2. Verify that `tuner_dashboard_Version21.py` exists in the same directory
    3. Look for any missing import statements in the traceback above
    4. Check for syntax errors or indentation issues in the main application file
    5. Ensure all dependencies specified in requirements.txt are compatible
    """)
    
    # Stop execution to prevent further errors
    st.stop()
