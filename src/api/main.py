# API will serve as a placeholder for now. It confirms that the web server component is working.

from fastapi import FastAPI

# Initialize the FastAPI app
app = FastAPI(
    title="Computer Vision Security System API",
    description="API for handling security alerts and attendance.",
    version="1.0.0"
)

@app.get("/", tags=["Root"])
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "success", "message": "Access Monitor API is running"}

@app.post("/alert", tags=["Alerts"])
def trigger_alert(details: dict):
    """A placeholder endpoint for handling alerts for unknown faces."""
    # In a future version, this could send an email or a Slack notification.
    print(f"INFO: Alert received for an unknown person. Details: {details}")
    return {"status": "alert logged successfully"}