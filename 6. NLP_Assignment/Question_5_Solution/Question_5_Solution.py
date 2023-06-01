import fasttext

# Load the pre-trained language detection model
model_path = 'data_to_utilise/lid.176.ftz'
model = fasttext.load_model(model_path)

# Define the text for language detection
text = "भाषा पहचान के लिए"

# Perform language detection
prediction = model.predict(text, k=1)  # k=1 for top prediction

# Extract the predicted language and confidence score
predicted_language = prediction[0][0].replace('__label__', '')
confidence_score = prediction[1][0]

# Print the predicted language and confidence score
print(f"Predicted Language: {predicted_language}")
print(f"Confidence Score: {confidence_score}")