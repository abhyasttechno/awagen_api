import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get('GEMINI_API_KEY')
client = None # Initialize as None

if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set!")
else:
    try:
        client = genai.Client(api_key=API_KEY)
        # Optional: Test configuration
        # logging.info("Gemini API client initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to configure Gemini API client: {e}")
        client = None # Ensure client is None if initialization fails


@app.route('/generate-post', methods=['POST'])
def generate_post():
    if not client:
        logging.error("Gemini API client is not initialized.")
        return jsonify({"error": "Server is not configured with the API key or failed to initialize API client."}), 500

    # List to keep track of successfully uploaded file objects from the File API
    uploaded_gemini_files = []

    try:
        # Get data from request.form (for text) and request.files (for images)
        post_type = request.form.get('postType', 'General')
        input_language = request.form.get('inputLanguage', 'English')
        output_language = request.form.get('outputLanguage', 'English')
        user_context = request.form.get('userContext') # Mandatory text input

        # Get the list of uploaded files from the request
        uploaded_files_from_request = request.files.getlist('images')

        # Basic validation for mandatory text context
        if not user_context or not user_context.strip():
            return jsonify({"error": "User context is required"}), 400

        logging.info(f"Received request: postType={post_type}, outputLanguage={output_language}, context='{user_context[:50]}...', images_count={len(uploaded_files_from_request)}")

        # --- MODIFIED: Upload files using the File API ---
        if len(uploaded_files_from_request) > 0:
            logging.info(f"Attempting to upload {len(uploaded_files_from_request)} files to Gemini File API...")
            for file_storage in uploaded_files_from_request:
                logging.info(f"Processing file from request: {file_storage.filename}, Detected MIME: {file_storage.mimetype}")
                if file_storage.filename and file_storage.mimetype and file_storage.mimetype.startswith('image/'):
                    try:
                        # FileStorage object behaves like a file, we can pass it directly
                        # The File API call uploads the data to Google's backend
                        gemini_file = client.files.upload(file=file_storage)
                        uploaded_gemini_files.append(gemini_file)
                        logging.info(f"Successfully uploaded file {file_storage.filename} to Gemini File API. URI: {gemini_file.uri}")
                    except Exception as e:
                        logging.error(f"Error uploading file {file_storage.filename} to Gemini File API: {e}")
                        logging.error(f"Full traceback for File API upload error of {file_storage.filename}:")
                        traceback.print_exc()
                        # Decide how to handle: pass skips the file, raise would stop the request
                        # For now, we'll just log and skip this problematic file
                        pass
                else:
                     logging.warning(f"Skipping non-image file from request: {file_storage.filename or 'No filename'}, MIME: {file_storage.mimetype or 'Unknown'}")

        # Check if at least text is available, or if images were intended
        if not user_context.strip() and not uploaded_gemini_files:
             logging.error("No user context and no valid images uploaded.")
             return jsonify({"error": "Please provide text details or upload valid images."}), 400

        # --- MODIFIED: Construct content parts using URIs ---
        # Start with the text part
        text_part_content = f"""Generate social media content based on the following requirements:
- Post Type: {post_type}
- Input Context Language: {input_language}
- Output Language: {output_language}
- User Context/Details: "{user_context.strip()}"
"""
        if len(uploaded_gemini_files) > 0:
             text_part_content += f"""
- Note: {len(uploaded_gemini_files)} image(s) have been provided via URIs. Analyze them along with the text context to generate the post content.
Do NOT explicitly describe the images in the text unless specifically requested in the user context. Focus on generating relevant post text based on the combined visual and text input."""
        else:
             text_part_content += """
No images were provided. Generate content solely based on the user context."""

        text_part_content += """
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

        # Create the list of parts, starting with the text part
        parts = [types.Part.from_text(text=text_part_content)]

        # Add image parts using URIs of the successfully uploaded files
        for gemini_file in uploaded_gemini_files:
             try:
                parts.append(types.Part.from_uri(file_uri=gemini_file.uri, mime_type=gemini_file.mime_type))
                logging.info(f"Added URI part for {gemini_file.uri}")
             except Exception as e:
                 logging.error(f"Error creating Part from URI {gemini_file.uri}: {e}")
                 traceback.print_exc()
                 # Continue adding other parts even if one URI part fails
                 pass

        # Check if there are any parts to send (at least the text part should be there)
        if not parts:
             logging.error("No parts (text or image URIs) successfully prepared for AI call.")
             # This case is unlikely if user_context is mandatory and handled first, but good safeguard
             return jsonify({"error": "Could not prepare content for the AI model."}), 500


        logging.info(f"Sending {len(parts)} parts (including {len(uploaded_gemini_files)} image URIs) to Gemini model...")

        # Call the Gemini API using the list of parts with URIs
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Content(role="user", parts=parts)]
        )
        # --- END MODIFIED ---

        generated_text = response.text
        logging.info(f"Received response text length: {len(generated_text)}")

        # Parse the generated text into sections (regex remains similar)
        facebook_match = re.search(r'### Facebook Post ###\s*([\s\S]*?)(?=\s*### X \(Twitter\) Post ###|$)', generated_text)
        x_match = re.search(r'### X \(Twitter\) Post ###\s*([\s\S]*?)(?=\s*### Instagram Post ###|$)', generated_text)
        instagram_match = re.search(r'### Instagram Post ###\s*([\s\S]*?)$', generated_text)

        facebook_content = facebook_match.group(1).strip() if facebook_match else ""
        x_content = x_match.group(1).strip() if x_match else ""
        instagram_content = instagram_match.group(1).strip() if instagram_match else ""

        # Check if any content was successfully parsed
        if not facebook_content and not x_content and not instagram_content:
            if generated_text and generated_text.strip():
                logging.warning("Failed to parse sections from AI response, returning raw text.")
                facebook_content = "Warning: Could not parse response into sections. The AI generated content might not be in the expected format.\n\nRaw AI Response:\n" + generated_text.strip()
                x_content = "N/A (Parsing failed)"
                instagram_content = "N/A (Parsing failed)"
            else:
                logging.error("Gemini generated no content.")
                return jsonify({"error": "Failed to generate content from the AI model. The response was empty."}), 500

        return jsonify({
            "facebook": facebook_content,
            "x": x_content,
            "instagram": instagram_content
        })

    except Exception as e:
        logging.error(f"An error occurred during processing or AI call: {e}", exc_info=True)
        error_message = str(e)
        if "API key" in error_message or "authentication" in error_message:
             error_message = "API authentication failed. Check your GEMINI_API_KEY."
        elif "quota" in error_message:
             error_message = "API quota exceeded. Please try again later."
        elif "rate limit" in error_message:
             error_message = "API rate limit exceeded. Please try again later."
        elif "invalid_argument" in error_message or "Bad Request" in error_message:
             error_message = f"Invalid input sent to AI model: {error_message}"
        else:
             error_message = f"An unexpected error occurred: {error_message}"

        return jsonify({"error": error_message}), 500




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
   
    app.run(host='0.0.0.0', port=port, debug=True)
