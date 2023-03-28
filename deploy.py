import streamlit as st
import numpy as np
#from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import pickle 

def predict_labels(input_text):
  # Convert text and labels to numerical vectors
    #words = input_text.split()
    #print(words)
    MAX_NUM_WORDS = 10000
    MAX_SEQUENCE_LENGTH = 20
    
    with open(r'C:\Users\MY-PC\OneDrive\Documents\GitHub\Named_Entity_Recognition\tokenizer.pkl', 'rb') as f:
         tokenizer = pickle.load(f)
    #tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(input_text.split())
    X = tokenizer.texts_to_sequences(input_text.split())
    #print(X)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    #print(X)
    

    label2idx = {'O': 0, 'S-Disease': 1, 'B-Disease': 2, 'B-Organism': 3, 'I-Organism': 4,
             'S-Chemical_Substance': 5, 'S-Protien': 6, 'S-Organism': 7, 'B-Chemical_Substance': 8,
             'B-Protien': 9, 'I-Disease': 10, 'B-Medication': 11, 'S-Medication': 12,
             'S-Anatomical_Substances': 13, 'I-Protien': 14, 'B-Anatomical_Substances': 15,
             'I-Chemical_Substance': 16, 'I-Anatomical_Substances': 17, 'I-Medication': 18,
             'S-Gene': 19, 'E-Disease': 20, 'E-Protien': 21, 'E-Organism': 22, 'S_Anatomical_Substances': 23,
             'B-Gene': 24, 'E-Medication': 25, 'E-Anatomical_Substances': 26, 'E-Chemical_Substance': 27,
             'I-Gene': 28, 'E-Gene': 29}
    
    # Load the trained model
    model = load_model(r'C:\Users\MY-PC\OneDrive\Documents\GitHub\Named_Entity_Recognition\model.h5')

    # Predict on the new data
    predictions = model.predict(X)

    # Convert predictions to labels
    predicted_labels = []
    for pred in predictions:
        #print(pred)
        #predicted_labels.append(list(label2idx.keys())[list(label2idx.values()).index(np.argmax(pred))]) 
        predicted_label_idx = np.argmax(pred)
        #print(predicted_label_idx)
        predicted_label = list(label2idx.keys())[predicted_label_idx]
        predicted_labels.append(predicted_label)
   
    return predicted_labels


#Create the Streamlit app
st.title('Named Entity Recognition')

input_text = st.text_area('Enter some text:')

st.write("OR")

option = st.selectbox(
    'Choose a text from dropdown: ',
    ('Malaria remains an important public health problem despite efforts to control it. Besides active transmission, relapsing malaria caused by dormant liver stages of Plasmodium vivax and Plasmodium ovale hypnozoites is a major hurdle in malaria control and elimination programs. Primaquine is the most widely used drug for radical cure of malaria.', 
     'RTS,S/AS01 (RTS,S) is the first and, to date, only vaccine that has demonstrated it can significantly reduce malaria in young children living in moderate-to-high malaria transmission areas. It acts against the Plasmodium falciparum parasite, the deadliest malaria parasite globally.', 
     'Malaria is an acute febrile illness caused by Plasmodium parasites, which are spread to people through the bites of infected female Anopheles mosquitoes. It is preventable and curable.'))

# Define the colors for each label
entity_colors = {"O": "white",
          "S-Disease": "#ffa07a",
          "B-Disease": "#ffa07a",
          "B-Organism": "#98fb98",
          "I-Organism": "#98fb98",
          "S-Chemical_Substance": "#1e90ff",
          "S-Protien": "#9370db",
          "S-Organism": "#98fb98",
          "B-Chemical_Substance": "#1e90ff",
          "B-Protien": "#9370db",
          "I-Disease": "#ffa07a",
          "B-Medication": "#ffff00",
          "S-Medication": "#ffff00",
          "S-Anatomical_Substances": "#ffb6c1",
          "I-Protien": "#9370db",
          "B-Anatomical_Substances": "#ffb6c1",
          "I-Chemical_Substance": "#1e90ff",
          "I-Anatomical_Substances": "#ffb6c1",
          "I-Medication": "#ffff00",
          "S-Gene": "#87cefa",
          "E-Disease": "#ffa07a",
          "E-Protien": "#9370db",
          "E-Organism": "#98fb98",
          "S_Anatomical_Substances": "#ffb6c1",
          "B-Gene": "#87cefa",
          "E-Medication": "#ffff00",
          "E-Anatomical_Substances": "#ffb6c1",
          "E-Chemical_Substance": "#1e90ff",
          "I-Gene": "#87cefa",
          "E-Gene": "#87cefa"}


if st.button('Predict'):
    if input_text:
        predicted_labels = predict_labels(input_text)
        # Split input text into words
        words = input_text.split()
        # Use a for loop to replace each word with highlighted text using the corresponding predicted label
        highlighted_text = ""
        for i, word in enumerate(words):
            label = predicted_labels[i]
            if label != 'O':
                word_color = entity_colors.get(label, "yellow")
                label_color = entity_colors.get(label + '-label', "black")
                highlighted_text += f'<mark style="background-color: {word_color}; color: {label_color}">{word} ({label})</mark> '
            else:
                highlighted_text += f'{word} '
        # Display the highlighted text
        st.markdown(highlighted_text, unsafe_allow_html=True)
    elif option:
        predicted_labels = predict_labels(option)
        # Split input text into words
        words = option.split()
        # Use a for loop to replace each word with highlighted text using the corresponding predicted label
        highlighted_text = ""
        for i, word in enumerate(words):
            label = predicted_labels[i]
            if label != 'O':
               word_color = entity_colors.get(label, "yellow")
               label_color = entity_colors.get(label + '-label', "black")
               highlighted_text += f'<mark style="background-color: {word_color}; color: {label_color}">{word} ({label})</mark> '
            else:
               highlighted_text += f'{word} '
        # Display the highlighted text
        st.markdown(highlighted_text, unsafe_allow_html=True)