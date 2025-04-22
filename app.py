import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Enable CORS for all origins. In a production environment, you might want to restrict this
# to only your frontend domain for better security.
CORS(app)

# Get API Key from environment variable
# This is the SECURE way to handle keys in Cloud Run
API_KEY = os.environ.get('GEMINI_API_KEY')

client = genai.Client(api_key=API_KEY)

if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set!")
    # In production, you might want to stop the app or return an error on startup
    # For this example, we'll proceed but API calls will fail
    pass # Or sys.exit(1)


# MODEL_ID = "gemini-1.5-flash-latest" # Use a suitable model, e.g., gemini-1.5-flash-latest
# Use the configured API key
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        # Optional: Test if configuration works (e.g., list models)
        # for m in genai.list_models():
        #     logging.info(f"Available model: {m.name}")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API: {e}")
        # Set API_KEY to None to prevent further attempts
        API_KEY = None


@app.route('/generate-post', methods=['POST'])
def generate_post():
    if not API_KEY:
        logging.error("API key is not configured.")
        return jsonify({"error": "Server is not configured with the API key."}), 500

    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Extract data from the request
        post_type = data.get('postType', 'General')
        input_language = data.get('inputLanguage', 'English')
        output_language = data.get('outputLanguage', 'English')
        user_context = data.get('userContext')
        images_selected_count = data.get('imagesSelectedCount', 0) # Frontend tells us how many images

        # Basic validation
        if not user_context or not user_context.strip():
            return jsonify({"error": "User context is required"}), 400

        # Construct the prompt (similar logic to frontend, but built here)
        prompt = f"""Generate social media content based on the following requirements:
- Post Type: {post_type}
- Input Context Language: {input_language}
- Output Language: {output_language}
- User Context/Details: "{user_context.strip()}"
"""

        if images_selected_count > 0:
            prompt += f"""- Note: The user has uploaded {images_selected_count} image(s). Consider how the post text could complement a visual element, or just be aware images will accompany the post. Do NOT describe the specific content of the uploaded images, and do NOT include image data in the response."""

        prompt += """
Please provide content specifically tailored for each platform below. Aim for the general conventions of each platform regarding length and style. Use clear headings for each section exactly as follows, followed by the generated text.
Also follow 5W1H approach which is what, who, when, where, why and how? in the post by analyzing the given information.
Dont mention 5W1H approach which is what, who, when, where, why and how? in the post directly.  :

### Facebook Post ###
[Generate Facebook content here. It can be a moderate length, suitable for a typical feed post. Include relevant hashtags.]

### X (Twitter) Post ###
[Generate X content here. Be concise, strictly adhere to a character limit appropriate for X (~280 chars including hashtags is a good target). Include relevant hashtags.]

### Instagram Post ###
[Generate Instagram caption here. It should be engaging and can use line breaks for readability, but aim for a concise to moderate length compared to Facebook. Include relevant hashtags.]

Ensure the language of the generated content is strictly in {output_language}.
"""

        logging.info(f"Sending prompt to Gemini: {prompt[:200]}...") # Log first 200 chars
        
        # Call the Gemini API
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
      

        logging.info(f"Received response from Gemini (partial): {str(response.text)[:200]}...") # Log partial response

        # Extract text from the response
        generated_text = response.text

        # Parse the generated text into sections
        # Use regex to find content between markers
        facebook_match = re.search(r'### Facebook Post ###\s*([\s\S]*?)(?=### X \(Twitter\) Post ###|$)', generated_text)
        x_match = re.search(r'### X \(Twitter\) Post ###\s*([\s\S]*?)(?=### Instagram Post ###|$)', generated_text)
        instagram_match = re.search(r'### Instagram Post ###\s*([\s\S]*?)$', generated_text)

        facebook_content = facebook_match.group(1).strip() if facebook_match else ""
        x_content = x_match.group(1).strip() if x_match else ""
        instagram_content = instagram_match.group(1).strip() if instagram_match else ""

        # Check if any content was successfully parsed
        if not facebook_content and not x_content and not instagram_content:
             # If parsing failed but we got *some* text, return it as a fallback
            if generated_text and generated_text.strip():
                logging.warning("Failed to parse sections, returning raw text.")
                facebook_content = "Warning: Could not parse response into sections.\n\nRaw AI Response:\n" + generated_text.strip()
                x_content = "N/A (Parsing failed)"
                instagram_content = "N/A (Parsing failed)"
            else:
                 # No content at all
                logging.error("Gemini generated no content.")
                return jsonify({"error": "Failed to generate content from the AI model."}), 500


        # Return the parsed content as JSON
        return jsonify({
            "facebook": facebook_content,
            "x": x_content,
            "instagram": instagram_content
        })

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        # Return a JSON error response
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

# Cloud Run uses the PORT environment variable
# For local testing, you can set a default like 8080
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080', debug=True) # debug=True for local dev, change to False for prod
