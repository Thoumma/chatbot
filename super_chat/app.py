from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Hugging Face client
# print("HF_TOKEN loaded:", bool(os.getenv("HF_TOKEN")))
client = InferenceClient(token=os.getenv("HF_TOKEN"))

chat_history = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global chat_history
    
    try:
        user_input = request.json["message"]
        chat_history.append({"role": "user", "content": user_input})

        # Build conversation context
        messages = [
            {"role": "system", "content": "You are JD's helpful assistant named Grandma Lara. Respond clearly and kindly in English, no matter the topic."}
        ] + chat_history

        # Use Hugging Face's free model
        response = client.chat_completion(
            messages=messages,
            model="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=500
        )

        reply = response.choices[0].message.content.strip()
        chat_history.append({"role": "assistant", "content": reply})

        return jsonify({"response": reply})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": "Sorry dear, I had a little trouble. Please try again!"}), 200

if __name__ == "__main__":
    app.run( port=5174, use_reloader=False)