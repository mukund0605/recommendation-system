from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.jpy.ipynb")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = request.form["input_data"]
        
        # Preprocess the input data (if needed)
        # Make predictions using the model
        prediction = model.predict([input_data])[0]
        
        # Pass the prediction to the result template
        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
