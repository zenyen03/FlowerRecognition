import requests
import os

# Get the full path to the image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "rose.jpg")

# Make sure the image exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    exit(1)

# Use Render URL
url = "https://flower-tqjv.onrender.com/predict"
print(f"Using URL: {url}")
print(f"Image path: {image_path}")

try:
    # Send the request
    with open(image_path, 'rb') as img:
        files = {
            'image': ('primrose.jpg', img, 'image/jpeg')
        }
        print("Sending request...")
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {str(e)}")
    print(f"Error type: {type(e).__name__}")
