# Core Pkgs
import streamlit as st
import altair as alt

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import joblib

# Import Model/Pipeline
pipe_lr = joblib.load(open("/Users/kamrankhanalwi/Desktop/emosence/App/Models/emotion_classifier2.pkl", "rb"))

# Functions
# Fxn to predict emotions 
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Fxn to get probability
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Emoji Dictonary
# emotions_emoji_dict = {'anger' : '😡', 'fear' : '😨', 'joy' : '😂', 'love' : '❤️', 'sadness' : '😔', 'surprise' : '😱'}

emotions_emoji_dict = {
                        'anger' : '😡', 
                        'boredom' : '😑', 
                        'disgust' : '🤮', 
                        'empty' : '🫙', 
                        'enthusiasm' : '🤩',
                        'fear' : '😨',
                        'fun': '🥳', 
                        'happiness' : '😄', 
                        'hate' : '🤬', 
                        'joy' : '😂', 
                        'love' : '❤️', 
                        'neutral' : '😐', 
                        'relief' : '😮‍💨',
                        'sad' : '😔', 
                        'shame' : '😅', 
                        'surprise' : '😱', 
                        'worry' : '🤔'
                    }

# emotions_emoji_dict = {
#     'anger' : '😡',
#     'disgust' : '🤮', 
#     'fear' : '😨',
#     'joy' : '😂',
#     'love' : '❤️',
#     'neutral' : '😐',
#     'sadness' : '😔',
#     'shame' : '😅', 
#     'surprise' : '😱'
#     }

def main(): 
    st.title("Emosence: Unveiling Emotions in Text")
    # st.title("Emotion Classifier App")
    menu = ["Home", "Classifier", "Know Your Emotion"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Classifier":
        st.subheader("Emotion Classifier")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

            if submit_text:
                col1,col2 = st.columns(2)

                # Apply Fxn Here
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)

                with col1:
                    st.success("Original Text")
                    st.write(raw_text)

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{} : {}".format(prediction, emoji_icon))
                    st.write("Confidence : {}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    # st.write(probability)
                    proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                    # st.write(proba_df.T)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ['emotions', 'probability']

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)
    elif choice == "Know Your Emotion":
        st.header("Know Your Emotions-Emoji")

        st.subheader("Love : ❤️")
        st.subheader("Anger : 😡")
        st.subheader("Happiness : 😄")
        st.subheader("Sad : 😔")
        st.subheader("Joy : 😂")
        st.subheader("Fear : 😨")
        st.subheader("Neutral : 😐")
        st.subheader("Worry : 🤔")
        st.subheader("Relief : 😮‍💨")
        st.subheader("Boredom : 😑")
        st.subheader("Fun : 🥳")
        st.subheader("Disgust : 🤮")
        st.subheader("Hate : 🤬")
        st.subheader("Empty : 🫙")
        st.subheader("Enthusiasm : 🤩")
        st.subheader("Shame : 😅")
        st.subheader("Surprise : 😱")

    else:
        # st.subheader("About")
        st.subheader("Welcome to Emosence, where the symphony of words resonates with emotions! 🌟")
        st.write("In the dynamic realm of digital communication, Emosence stands as a beacon of emotional intelligence, bringing forth the power to decipher and amplify the emotional nuances within written expressions. This cutting-edge app seamlessly integrates natural language processing and machine learning to decode the sentiments behind your words.")
        
        st.header("Key Features:")

        st.subheader("📚 Text-to-Emotion Transformation: ")
        st.write("Embark on a journey where your words transcend the ordinary. Emosence employs advanced machine learning algorithms, including the enchanting capabilities of scikit-learn's CountVectorizer, to analyze and illuminate the emotional palette within your text.")

        st.subheader("🧠 Intelligent Emotion Recognition:")
        st.write("Delve into the rich tapestry of emotions. Emosence utilizes sophisticated natural language processing techniques to precisely recognize a spectrum of emotions, from exuberant joy to poignant sadness.")

        st.subheader("🔄 Real-Time Interaction:")
        st.write("Experience the magic of emotions evolving in real-time as you type or modify your text. Emosence provides instantaneous feedback, allowing you to witness the fluidity of emotions within your words.")

        st.subheader("🌐 Versatility Across Text Genres:")
        st.write("From heartfelt messages to professional communications, Emosence seamlessly adapts to diverse text genres. Its versatility ensures a nuanced emotional analysis across different contexts and writing styles.")


if __name__ == '__main__':
    main()