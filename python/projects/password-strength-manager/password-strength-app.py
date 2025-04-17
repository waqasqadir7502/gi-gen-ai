import re 
import streamlit as st
import random


# title 
st.title(" 🔐 Password Strength Checker/Meter")

# Checking Password Strength
def check_password_strength(password):
    score = 0
    feedback = []
    blacklist = ["passwords", "password123", "abcdefg", "12345678", "admin", "qwerty" "tryme"]

    # Blacklist passwords check
    if password.lower() in blacklist:
        feedback.append("❌ Generate a Unique Password")
        return 1, feedback

    # Length Check

    if len(password) >= 8:
        score += 1
    else:
        feedback.append("❌ Password Must Atleast 8 Characters or more")

    # Words (Uppercase and Lowercase) Check

    if re.search(r"[A-Z]", password) and re.search(r"[a-z]", password):
        score += 1
    else:
        feedback.append("❌ Password Must Inclucde Both Upper and lowercase letters")

    # Digit Check

    if re.search(r"\d", password):
        score += 1
    else:
        feedback.append("❌ Password Must contain atleast one digit 0-9")

    # Special Character Check

    if re.search(r"[!@#$%^&*]", password):
        score += 1
    else:
        feedback.append("❌ Password Must contain atleast one special character")

    return score, feedback
    


# Getting user input
password = st.text_input(" Enter Your Password" , type="password")

if password:
    score , feedback = check_password_strength(password)

    st.markdown("📝 Results")
        # Strength Rate

    if score == 4:
        st.success("✅ Password is Strong!")
    elif score == 3:
        st.warning("⚠ Password is Moderate - Consider making it more secure")
    else:
        st.error("❌ Password is weak - Improvement Required!")

    for msg in feedback:
        st.write(msg)


# Generate Strong Password 
def generate_strong_pass(lenght = 12):
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lower = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    special_char = "!@#$%^&*"
    all_chars = upper + lower+ digits + special_char

    password = [
        random.choice(upper),
        random.choice(lower),
        random.choice(digits),
        random.choice(special_char),
    ]

    password += random.choices(all_chars, k= lenght - 4)
    random.shuffle(password)
    return "".join(password)


if st.button("Generate Strong Password"):
    strong_pass = generate_strong_pass()
    st.code(strong_pass, language="text")
