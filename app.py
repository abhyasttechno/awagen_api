import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types # Import types for Part
import logging
# No need for `io` if using file objects directly from request.files

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Enable CORS for all origins. In a production environment, you might want to restrict this
# to only your frontend domain for better security.
CORS(app)

# Get API Key from environment variable
API_KEY = os.environ.get('GEMINI_API_KEY')

# Initialize client only if API_KEY is available
client = None
if not API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set!")
    # API calls will fail later. Keep client as None to handle this state.
else:
    try:
        client = genai.Client(api_key=API_KEY)
        # Optional: Test if configuration works (e.g., list models)
        # for m in genai.list_models():
        #     logging.info(f"Available model: {m.name}") # Uncomment to test client setup
    except Exception as e:
        logging.error(f"Failed to configure Gemini API client: {e}")
        # Set client back to None if initialization failed
        client = None


@app.route('/generate-post', methods=['POST'])
def generate_post():
    # Check if API key and client were successfully configured
    if not client:
        logging.error("Gemini API client is not initialized.")
        return jsonify({"error": "Server is not configured with the API key or failed to initialize API client."}), 500

    try:
        # --- MODIFIED: Get data from request.form (for text) and request.files (for images) ---
        # Use request.form for text fields in multipart/form-data
        post_type = request.form.get('postType', 'General')
        input_language = request.form.get('inputLanguage', 'English')
        output_language = request.form.get('outputLanguage', 'English')
        user_context = request.form.get('userContext') # Mandatory text input

        # Use request.files.getlist('images') to get the list of uploaded files
        uploaded_files = request.files.getlist('images')
        # --- END MODIFIED ---

        # Basic validation for mandatory text context
        if not user_context or not user_context.strip():
            return jsonify({"error": "User context is required"}), 400

        logging.info(f"Received request: postType={post_type}, outputLanguage={output_language}, context='{user_context[:50]}...', images_count={len(uploaded_files)}")

        # --- MODIFIED: Construct content parts for the API call ---
        # The 'contents' parameter for generate_content is a list of types.Content objects.
        # For a single turn (user query), this list usually has one entry.
        # The 'parts' within that types.Content object is a list containing text and image parts.

        # Start with the text part containing the instructions and user context
        # We include the instructions directly in the text part now, as the AI
        # receives the images alongside this text part.
        text_part_content = f"""Generate social media content based on the following requirements:
- Post Type: {post_type}
- Input Context Language: {input_language}
- Output Language: {output_language}
- User Context/Details: "{user_context.strip()}"
"""

        if len(uploaded_files) > 0:
            text_part_content += """
Analyze the provided image(s) along with the user context to generate the content.
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

        # Add image parts from uploaded files
        for file in uploaded_files:
            # Ensure the file has a name, type, and is an image
            if file.filename and file.mimetype and file.mimetype.startswith('image/'):
                try:
                    # Read the file content bytes
                    # file is a FileStorage object which behaves like a file
                    image_bytes = file.read()
                    parts.append(types.Part.from_data(data=image_bytes, mime_type=file.mimetype))
                    logging.info(f"Successfully added image part for: {file.filename}, MIME: {file.mimetype}")
                except Exception as file_read_error:
                    logging.error(f"Error reading uploaded file {file.filename}: {file_read_error}")
                    # Optionally, return an error to the user or skip the file
                    # For now, we'll just log and skip
                    pass
            else:
                logging.warning(f"Skipping invalid file upload: {file.filename or 'No filename'}, MIME: {file.mimetype or 'Unknown'}")


        # Check if there are any parts to send
        if not parts:
             logging.error("No valid parts (text or images) to send to AI model.")
             return jsonify({"error": "No valid content provided to generate post."}), 400

        logging.info(f"Sending {len(parts)} parts to Gemini model...")

        # Call the Gemini API with the list of parts
        # Use generate_content for a single interaction turn, not stream
        # contents is a list of Content objects. For a single turn, it's [user_content]
        response = client.models.generate_content(
            model="gemini-2.0-flash", # You can also use gemini-1.5-flash-latest if preferred
            contents=[types.Content(role="user", parts=parts)]
        )

        # --- END MODIFIED ---

        # Extract text from the response
        generated_text = response.text
        logging.info(f"Received response text length: {len(generated_text)}")

        # Parse the generated text into sections (regex remains similar)
        # Use regex to find content between markers
        # Added optional whitespace/newline handling around markers and content
        facebook_match = re.search(r'### Facebook Post ###\s*([\s\S]*?)(?=\s*### X \(Twitter\) Post ###|$)', generated_text)
        x_match = re.search(r'### X \(Twitter\) Post ###\s*([\s\S]*?)(?=\s*### Instagram Post ###|$)', generated_text)
        instagram_match = re.search(r'### Instagram Post ###\s*([\s\S]*?)$', generated_text)

        facebook_content = facebook_match.group(1).strip() if facebook_match else ""
        x_content = x_match.group(1).strip() if x_match else ""
        instagram_content = instagram_match.group(1).strip() if instagram_match else ""

        # Check if any content was successfully parsed
        if not facebook_content and not x_content and not instagram_content:
             # If parsing failed but we got *some* text, return it as a fallback
            if generated_text and generated_text.strip():
                logging.warning("Failed to parse sections from AI response, returning raw text.")
                facebook_content = "Warning: Could not parse response into sections. The AI generated content might not be in the expected format.\n\nRaw AI Response:\n" + generated_text.strip()
                x_content = "N/A (Parsing failed)"
                instagram_content = "N/A (Parsing failed)"
            else:
                 # No content at all
                logging.error("Gemini generated no content.")
                return jsonify({"error": "Failed to generate content from the AI model. The response was empty."}), 500


        # Return the parsed content as JSON
        return jsonify({
            "facebook": facebook_content,
            "x": x_content,
            "instagram": instagram_content
        })

    except Exception as e:
        logging.error(f"An error occurred during processing or AI call: {e}", exc_info=True)
        # Return a JSON error response
        # Include a more user-friendly message
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

# Cloud Run uses the PORT environment variable
# For local testing, you can set a default like 8080
if __name__ == '__main__':
    # Use provided PORT or default to 8080
    port = int(os.environ.get('PORT', 8080))
 
    app.run(host='0.0.0.0', port=port, debug=True)
