import os
from flask import Flask, render_template, request
from eo import get_satellite_image

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            latitude = float(request.form["latitude"])
            longitude = float(request.form["longitude"])

            # Direct call to fetch the single best/latest 10m image
            result = get_satellite_image(latitude, longitude)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    # Dynamically bind to the PORT environment variable assigned by cloud hosts (like Render)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)