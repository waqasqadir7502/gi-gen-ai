# Importing Required Libararies
import streamlit as st
import hashlib 
import json
import os
from cryptography.fernet import Fernet

# Intitialize Sessions State
if "encryption_key"  not in st.session_state:
    st.session_state.encryption_key = Fernet.generate_key()
    st.session_state.cipher = Fernet(st.session_state.encryption_key)
    st.session_state.stored_data = {}
    st.session_state.failed_attempts = 0

# File path to store encrypted data
data_file = "stored_data.json"

# Load Data from JSON file
def load_data():
    if os.path.exists(data_file):
        with open(data_file, "r") as f:
            return json.load(f)
    return {}

# Save Data to JSON file 
def save_data(data):
    with open(data_file, "w") as f:
        json.dump(data, f)

# In memory Data storage
stored_data = load_data()
failed_attempts = 0

# Function for hashing the pass key
def hash_passkey(passkey):
    return hashlib.sha256(passkey.encode()).hexdigest()

# Function for Data Encryption 
def encrypt_data(text):
    return st.session_state.cipher.encrypt(text.encode()).decode()

# Function for Decrypting Data
def decrypt_data(encrypted_text):
   return st.session_state.cipher.decrypt(encrypted_text.encode()).decode()

# Function for Reseting Failed Attempts 
def reset_failed_attempts():
    st.session_state.failed_attempts = 0

# StreamLit UI 
st.title("🛡 Secure Data Encryption System")

# Navigation
menu = ["Home", "Store Data", "Retrieve Data", "Login"]
choice = st.sidebar.radio("🔽 Menu", menu)

# Pages
if choice == "Home":
    st.subheader("🏠 Welcome to Secure Data Encryption System")
    st.write("Use this app to **securely store and retrieve data** using unique passkeys")

elif choice == "Store Data":
    st.subheader("📁 Store Your Data Securely with Encryption")
    data = st.text_area("Enter Data: ")
    passkey = st.text_input("Enter your unique passkey", type="password")

    if st.button("Encrypt and Save"):
        if data and passkey:
            hashed = hash_passkey(passkey)
            encrypted_text = encrypt_data(data)
            st.session_state.stored_data[encrypted_text] = {"encrypted_text": encrypted_text, "passkey" : hashed}
            save_data(stored_data)
            st.success("✅ Your Data has been Encrypted and Securely Stored! ")
        else:
            st.error("⚠ Both Fields are required")

elif choice == "Retrieve Data":
    st.subheader("📂 Retrieve Your Securely Saved and Encrypted Data")
    encrypted_text = st.text_input("Enter Encrypted Data: ")
    passkey = st.text_input("Enter Your Unique Passkey: " , type="password")


    if st.session_state.failed_attempts >= 3:
        st.warning("🔒 Too many attempts! Redirecting to Login Page")
        st.rerun()

    if st.button("Decrypt"):
        if encrypted_text and passkey:
            decrypted_text = decrypt_data(encrypted_text)

            if decrypted_text:
                st.success(f"✅ Decrypted Data: {decrypted_text}")
            else:
                st.error(f"❌ Incorrect passkey! Attempts remaining: {3 - failed_attempts}")

                if failed_attempts >= 3:
                    st.warning("🔒 Too many failed attempts! Redirecting to Login Page.")
                    st.experimental_rerun()

        else:
            st.error("❌ Both Fields are required!")

elif choice == "Login":
    st.subheader("🔑 Reauthorization Required")
    login_pass = st.text_input("Enter Master Passkey/Password: ", type="password")

    if st.button("Login"):
        if login_pass == "admin123": #for demo purpose
            failed_attempts = 0
            st.success("✅ Reauthorized Successfully! Redirecting to Retrieve Data...")
            st.rerun()
        else:
            st.error("❌ Invalid Password")
