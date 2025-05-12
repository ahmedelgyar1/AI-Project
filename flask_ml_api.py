from flask import Flask, request, jsonify
from flasgger import Swagger
from real_model import predict_home_automation, load_models

app = Flask(__name__)

# Load models once at startup
if not load_models():
    raise Exception("Failed to load models.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # تأكد من وجود كل المدخلات المطلوبة
        required_fields = ["mood", "person_condition", "time_of_day", "at_home", "is_holiday"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # استدعاء دالة التنبؤ
        recommendations = predict_home_automation(
            mood=data["mood"],
            person_condition=data["person_condition"],
            time_of_day=data["time_of_day"],
            at_home=int(data["at_home"]),
            is_holiday=int(data["is_holiday"])
        )

        if recommendations is None:
            return jsonify({"error": "Prediction failed."}), 500

        return jsonify({
            "status": "success",
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Smart Home ML API is running."})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # لو على localhost هيشتغل عادي برضه
    app.run(host="0.0.0.0", port=port)