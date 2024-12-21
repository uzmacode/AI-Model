import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Initialize emotion analysis model using HuggingFace transformers
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None)

# Title of the app
st.title("Emotion Analysis Based on Text")

# Add custom CSS to change background color, text color, and button color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;  /* Light blue background */
        color: #333333;  /* Dark text color */
    }
    
    .stButton>button {
        background-color: #1f3c73;  /* Dark blue background for the button */
        color: white;  /* White text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #365f99;  /* Slightly lighter blue when hovered */
    }

    .streamlit-expanderHeader {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True
)

# Add a description
st.write("""
    This app uses a pre-trained model to analyze the emotion expressed in a piece of text.
    Just type in your text and hit the button to get the emotion analysis!
""")

# Text input from user
user_input = st.text_area("Enter text for emotion analysis:")

# Button to trigger emotion analysis
if st.button("Analyze Emotion"):
    if user_input:
        # Perform emotion analysis
        result = emotion_classifier(user_input)
        
        # Extract emotion and score
        emotion = result[0]['label']
        score = result[0]['score']

        # Display results
        st.write(f"**Detected Emotion:** {emotion}")
        st.write(f"**Confidence Score:** {score:.2f}")

        # Display visualization (optional)
        emotions = [res['label'] for res in result]
        scores = [res['score'] for res in result]

        # Plot the emotion probabilities
        df = pd.DataFrame({'Emotion': emotions, 'Score': scores})
        df = df.sort_values(by='Score', ascending=False)

        fig, ax = plt.subplots()
        ax.barh(df['Emotion'], df['Score'], color='skyblue')
        ax.set_xlabel('Probability')
        ax.set_title('Emotion Probabilities')

        st.pyplot(fig)
    else:
        st.warning("Please enter some text to analyze!")
