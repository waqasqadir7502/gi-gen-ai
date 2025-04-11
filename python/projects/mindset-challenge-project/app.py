#Importing Packages 

import streamlit as st
import pandas as pd 
import os
from io import BytesIO

#Setup Streamlit

st.set_page_config(page_title="Data Sweeper 💿", layout="wide")
st.title("Data Sweeper 💿")
st.write("Transform your files between CSV and Excel formats with built-in data cleaning and visualization")

upload_files = st.file_uploader("⬆ Upload your desired files (CSV and Excel Only!):", accept_multiple_files=True, type=["csv", "xlsx"] )

if upload_files:
    for file in upload_files:
        file_ext = os.path.splitext(file.name)[-1].lower()

        if file_ext == ".csv":
            df = pd.read_csv(file)
        elif file_ext == ".xlsx":
            df = pd.read_excel(file)
        else:
            st.error(f"⚠ Unsupported File Type: {file_ext}") 
            continue

        # Displaying the file name and size 
        st.write(f"File Name: {file.name}")
        st.write(f"File Size: {file.size/1024}")

        # Show 5 Row of our DF
        st.write(f"🔍 Preview the header of the Data Frame")
        st.dataframe(df.head())

        # Data Cleaning Options 
        st.subheader("🧹 Data Cleaning Options")
        if st.checkbox(f"Clean data for {file.name}"):
            col1,col2 = st.columns(2)

            with col1:
                if st.button(f"Remove duplicate from {file.name}"):
                    df.drop_duplicates(inplace=True)
                    st.write("Duplicate Remove!") 

            with col2:
                if st.button(f"Fill the missing values in {file.name}"):
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    st.write("Missing Values have been fullfilled!")

        # Choose Specific Data to keep or convert
        st.subheader("🎯 Select Columns to convert")
        columns = st.multiselect(f"Choose Columns for {file.name}", df.columns, default=df.columns)
        df = df[columns]

        # Create some visualisation
        st.subheader("📊 Data Visualization")
        if st.checkbox(f"Show Visualization for {file.name}"):
            st.bar_chart(df.select_dtypes(include="number").iloc[:, :2])

        # Conversion File in CSV or Excel
        st.subheader("File Conversion Options")
        conversion_types = st.radio(f"Convert {file.name} to: ", ["CSV","Excel"], key=file.name)
        if st.button(f"🔁Convert {file.name}"):
            buffer = BytesIO()
            if conversion_types == 'CSV':
                df.to_csv(buffer, index=False)
                file_name = file.name.replace(file_ext, ".csv")
                mime_type = "text/csv"
            elif conversion_types == "Excel":
                df.to_excel(buffer, index=False)
                file_name = file.name.replace(file_ext, ".xlsx")
                mime_type = "application/vnd.openxmlformats-officedoucments.spreadsheetml.sheet"
            buffer.seek(0)

            # Download Button 
            st.download_button(
                label=f"⬇ Download Your {file.name} as {conversion_types}",
                data=buffer,
                file_name=file_name,
                mime=mime_type
            )

        st.success("🎉 All Operations were successfull")