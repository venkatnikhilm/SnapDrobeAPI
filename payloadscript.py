import os
import base64
import requests

# CONFIGURATION
directory = "/Users/venkatnikhilm/Desktop/Projects/test_images"  # <-- change this to your images folder
endpoint = "http://0.0.0.0:8002/add_image"  # <-- change this to your endpoint

# Loop through each file in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)

    # Make sure it's a file (you can also check for image extensions if you want)
    if os.path.isfile(filepath):
        print(f"Processing {filename}...")

        # Open the image file and encode it
        with open(filepath, "rb") as image_file:
            image_bytes = image_file.read()
            encoded_string = base64.b64encode(image_bytes).decode('utf-8')

        # Create the body
        payload = {
            "image_b64": encoded_string
        }

        # Send POST request
        response = requests.post(endpoint, json=payload)

        # Check response
        if response.status_code == 200:
            print(f"Successfully sent {filename}")
        else:
            print(f"Failed to send {filename}: {response.status_code} - {response.text}")