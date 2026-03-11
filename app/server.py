# server.py

from flask import Flask, request, send_file, jsonify, Response
from flask_swagger_ui import get_swaggerui_blueprint
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
import os
import traceback
import json
import base64

from config import DEFAULT_CONFIGS
from handle_text import prepare_tts_input_with_context
from tts_handler import generate_speech, generate_speech_stream, get_models_formatted, get_voices, get_voices_formatted
from utils import getenv_bool, require_api_key, AUDIO_FORMAT_MIME_TYPES, DETAILED_ERROR_LOGGING

app = Flask(__name__)
load_dotenv()

# Swagger UI
SWAGGER_URL = '/docs'
API_URL = '/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint)

API_KEY = os.getenv('API_KEY', DEFAULT_CONFIGS["API_KEY"])
PORT = int(os.getenv('PORT', str(DEFAULT_CONFIGS["PORT"])))

DEFAULT_VOICE = os.getenv('DEFAULT_VOICE', DEFAULT_CONFIGS["DEFAULT_VOICE"])
DEFAULT_RESPONSE_FORMAT = os.getenv('DEFAULT_RESPONSE_FORMAT', DEFAULT_CONFIGS["DEFAULT_RESPONSE_FORMAT"])
DEFAULT_SPEED = float(os.getenv('DEFAULT_SPEED', str(DEFAULT_CONFIGS["DEFAULT_SPEED"])))
DEFAULT_LANGUAGE = os.getenv('DEFAULT_LANGUAGE', DEFAULT_CONFIGS["DEFAULT_LANGUAGE"])

REMOVE_FILTER = getenv_bool('REMOVE_FILTER', DEFAULT_CONFIGS["REMOVE_FILTER"])
EXPAND_API = getenv_bool('EXPAND_API', DEFAULT_CONFIGS["EXPAND_API"])

# DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'tts-1')

# Currently in "beta" — needs more extensive testing where drop-in replacement warranted
def generate_sse_audio_stream(text, voice, speed):
    """Generator function for SSE streaming with JSON events."""
    try:
        # Generate streaming audio chunks and convert to SSE format
        for chunk in generate_speech_stream(text, voice, speed):
            # Base64 encode the audio chunk
            encoded_audio = base64.b64encode(chunk).decode('utf-8')
            
            # Create SSE event for audio delta
            event_data = {
                "type": "speech.audio.delta",
                "audio": encoded_audio
            }
            
            # Format as SSE event
            yield f"data: {json.dumps(event_data)}\n\n"
        
        # Send completion event
        completion_event = {
            "type": "speech.audio.done",
            "usage": {
                "input_tokens": len(text.split()),  # Rough estimate
                "output_tokens": 0,  # Edge TTS doesn't provide this
                "total_tokens": len(text.split())
            }
        }
        yield f"data: {json.dumps(completion_event)}\n\n"
        
    except Exception as e:
        print(f"Error during SSE streaming: {e}")
        # Send error event
        error_event = {
            "type": "error",
            "error": str(e)
        }
        yield f"data: {json.dumps(error_event)}\n\n"

# OpenAI endpoint format
@app.route('/v1/audio/speech', methods=['POST'])
@app.route('/audio/speech', methods=['POST'])  # Add this line for the alias
@require_api_key
def text_to_speech():
    try:
        data = request.json
        if not data or 'input' not in data:
            return jsonify({"error": "Missing 'input' in request body"}), 400

        text = data.get('input')

        if not REMOVE_FILTER:
            text = prepare_tts_input_with_context(text)

        # model = data.get('model', DEFAULT_MODEL)
        language = data.get('language', DEFAULT_LANGUAGE)
        voice_from_request = data.get('voice')
        if voice_from_request:
            # Explicit voice always wins
            voice = voice_from_request
        else:
            # Auto-select first available voice for the requested language
            available_voices = get_voices(language)
            if available_voices:
                voice = available_voices[0].get('name', DEFAULT_VOICE)
            else:
                voice = DEFAULT_VOICE
        response_format = data.get('response_format', DEFAULT_RESPONSE_FORMAT)
        speed = float(data.get('speed', DEFAULT_SPEED))
        
        # Check stream format - only "sse" triggers streaming
        stream_format = data.get('stream_format', 'audio')  # 'audio' (default) or 'sse'
        
        mime_type = AUDIO_FORMAT_MIME_TYPES.get(response_format, "audio/mpeg")
        
        if stream_format == 'sse':
            # Return SSE streaming response with JSON events
            def generate_sse():
                for event in generate_sse_audio_stream(text, voice, speed):
                    yield event
            
            return Response(
                generate_sse(),
                mimetype='text/event-stream',
                headers={
                    'Content-Type': 'text/event-stream',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'  # Disable nginx buffering
                }
            )
        else:
            # Return raw audio data (like OpenAI) - can be piped to ffplay
            output_file_path = generate_speech(text, voice, response_format, speed)
            
            # Read the file and return raw audio data
            with open(output_file_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up the temporary file
            try:
                os.unlink(output_file_path)
            except OSError:
                pass  # File might already be cleaned up
            
            return Response(
                audio_data,
                mimetype=mime_type,
                headers={
                    'Content-Type': mime_type,
                    'Content-Length': str(len(audio_data)),
                    'Content-Disposition': f'attachment; filename="speech.{response_format}"'
                }
            )
            
    except Exception as e:
        if DETAILED_ERROR_LOGGING:
            app.logger.error(f"Error in text_to_speech: {str(e)}\n{traceback.format_exc()}")
        else:
            app.logger.error(f"Error in text_to_speech: {str(e)}")
        # Return a 500 error for unhandled exceptions, which is more standard than 400
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500

# OpenAI endpoint format
@app.route('/v1/models', methods=['GET', 'POST'])
@app.route('/models', methods=['GET', 'POST'])
@app.route('/v1/audio/models', methods=['GET', 'POST'])
@app.route('/audio/models', methods=['GET', 'POST'])
def list_models():
    return jsonify({"models": get_models_formatted()})

# OpenAI endpoint format
@app.route('/v1/audio/voices', methods=['GET', 'POST'])
@app.route('/audio/voices', methods=['GET', 'POST'])
def list_voices_formatted():
    return jsonify({"voices": get_voices_formatted()})

@app.route('/health', methods=['GET'])
@app.route('/healthz', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/swagger.json', methods=['GET'])
def swagger_spec():
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "OpenAI Edge TTS API",
            "description": "OpenAI-compatible Text-to-Speech API powered by Microsoft Edge TTS.",
            "version": "1.0.0"
        },
        "servers": [{"url": f"http://localhost:{PORT}"}],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer"
                }
            },
            "schemas": {
                "SpeechRequest": {
                    "type": "object",
                    "required": ["input"],
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "Text to synthesize.",
                            "example": "Hello, world!"
                        },
                        "voice": {
                            "type": "string",
                            "description": "Edge-TTS voice name or OpenAI voice alias (e.g. alloy, shimmer, en-US-AvaNeural). When omitted, a voice is auto-selected based on `language`.",
                            "example": "en-US-AvaNeural"
                        },
                        "language": {
                            "type": "string",
                            "description": "BCP-47 locale used to auto-select a voice when `voice` is not provided. Ignored when `voice` is explicitly set.",
                            "example": "en-US"
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
                            "default": "mp3",
                            "description": "Audio output format."
                        },
                        "speed": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 2,
                            "default": 1.0,
                            "description": "Playback speed multiplier (1.0 = normal)."
                        },
                        "stream_format": {
                            "type": "string",
                            "enum": ["audio", "sse"],
                            "default": "audio",
                            "description": "Response delivery mode. `audio` returns raw binary; `sse` streams Server-Sent Events."
                        }
                    }
                },
                "ElevenLabsRequest": {
                    "type": "object",
                    "required": ["text"],
                    "properties": {
                        "text": {"type": "string", "example": "Hello from ElevenLabs endpoint"}
                    }
                }
            }
        },
        "security": [{"BearerAuth": []}],
        "paths": {
            "/v1/audio/speech": {
                "post": {
                    "tags": ["Speech"],
                    "summary": "Generate speech audio",
                    "description": "Converts input text to audio. Pass `language` to auto-select a voice for that locale, or pass `voice` explicitly to override.",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SpeechRequest"}}}
                    },
                    "responses": {
                        "200": {
                            "description": "Audio file or SSE stream",
                            "content": {
                                "audio/mpeg": {"schema": {"type": "string", "format": "binary"}},
                                "text/event-stream": {"schema": {"type": "string"}}
                            }
                        },
                        "400": {"description": "Bad request"},
                        "401": {"description": "Unauthorized"},
                        "500": {"description": "Internal server error"}
                    }
                }
            },
            "/v1/models": {
                "get": {
                    "tags": ["Models"],
                    "summary": "List available TTS models",
                    "security": [],
                    "responses": {"200": {"description": "List of models", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/v1/audio/voices": {
                "get": {
                    "tags": ["Voices"],
                    "summary": "List OpenAI-compatible voices",
                    "security": [],
                    "responses": {"200": {"description": "List of OpenAI-mapped voices", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/v1/voices": {
                "get": {
                    "tags": ["Voices"],
                    "summary": "List edge-tts voices filtered by language",
                    "parameters": [
                        {
                            "name": "language",
                            "in": "query",
                            "description": "BCP-47 locale to filter voices (e.g. fr-FR). Omit for default language.",
                            "schema": {"type": "string"}
                        }
                    ],
                    "responses": {"200": {"description": "Filtered voice list", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/v1/voices/all": {
                "get": {
                    "tags": ["Voices"],
                    "summary": "List all available edge-tts voices",
                    "responses": {"200": {"description": "All voices", "content": {"application/json": {"schema": {"type": "object"}}}}}
                }
            },
            "/health": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Health check",
                    "security": [],
                    "responses": {"200": {"description": "OK", "content": {"application/json": {"schema": {"type": "object", "properties": {"status": {"type": "string", "example": "ok"}}}}}}}
                }
            },
            "/elevenlabs/v1/text-to-speech/{voice_id}": {
                "post": {
                    "tags": ["ElevenLabs (beta)"],
                    "summary": "ElevenLabs-compatible TTS endpoint",
                    "description": "Requires `EXPAND_API=true`.",
                    "parameters": [
                        {
                            "name": "voice_id",
                            "in": "path",
                            "required": True,
                            "description": "Edge-TTS voice name (e.g. en-US-AvaNeural)",
                            "schema": {"type": "string"}
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ElevenLabsRequest"}}}
                    },
                    "responses": {
                        "200": {"description": "MP3 audio file", "content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}}},
                        "400": {"description": "Bad request"},
                        "500": {"description": "Endpoint disabled or TTS error"}
                    }
                }
            },
            "/azure/cognitiveservices/v1": {
                "post": {
                    "tags": ["Azure (beta)"],
                    "summary": "Azure Cognitive Services-compatible SSML endpoint",
                    "description": "Accepts an SSML payload. Requires `EXPAND_API=true`.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/ssml+xml": {
                                "schema": {"type": "string"},
                                "example": "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='en-US-AvaNeural'>Hello</voice></speak>"
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "MP3 audio file", "content": {"audio/mpeg": {"schema": {"type": "string", "format": "binary"}}}},
                        "400": {"description": "Invalid SSML"},
                        "500": {"description": "Endpoint disabled or TTS error"}
                    }
                }
            }
        }
    }
    return jsonify(spec)

@app.route('/v1/voices', methods=['GET', 'POST'])
@app.route('/voices', methods=['GET', 'POST'])
@require_api_key
def list_voices():
    specific_language = None

    data = request.args if request.method == 'GET' else request.json
    if data and ('language' in data or 'locale' in data):
        specific_language = data.get('language') if 'language' in data else data.get('locale')

    return jsonify({"voices": get_voices(specific_language)})

@app.route('/v1/voices/all', methods=['GET', 'POST'])
@app.route('/voices/all', methods=['GET', 'POST'])
@require_api_key
def list_all_voices():
    return jsonify({"voices": get_voices('all')})

"""
Support for ElevenLabs and Azure AI Speech
    (currently in beta)
"""

# http://localhost:5050/elevenlabs/v1/text-to-speech
# http://localhost:5050/elevenlabs/v1/text-to-speech/en-US-AndrewNeural
@app.route('/elevenlabs/v1/text-to-speech/<voice_id>', methods=['POST'])
@require_api_key
def elevenlabs_tts(voice_id):
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # Parse the incoming JSON payload
    try:
        payload = request.json
        if not payload or 'text' not in payload:
            return jsonify({"error": "Missing 'text' in request body"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON payload: {str(e)}"}), 400

    text = payload['text']

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    voice = voice_id  # ElevenLabs uses the voice_id in the URL

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED  # Optional customization via payload.get('speed', DEFAULT_SPEED)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

# tts.speech.microsoft.com/cognitiveservices/v1
# https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
# http://localhost:5050/azure/cognitiveservices/v1
@app.route('/azure/cognitiveservices/v1', methods=['POST'])
@require_api_key
def azure_tts():
    if not EXPAND_API:
        return jsonify({"error": f"Endpoint not allowed"}), 500
    
    # Parse the SSML payload
    try:
        ssml_data = request.data.decode('utf-8')
        if not ssml_data:
            return jsonify({"error": "Missing SSML payload"}), 400

        # Extract the text and voice from SSML
        from xml.etree import ElementTree as ET
        root = ET.fromstring(ssml_data)
        text = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').text
        voice = root.find('.//{http://www.w3.org/2001/10/synthesis}voice').get('name')
    except Exception as e:
        return jsonify({"error": f"Invalid SSML payload: {str(e)}"}), 400

    # Use default settings for edge-tts
    response_format = 'mp3'
    speed = DEFAULT_SPEED

    if not REMOVE_FILTER:
        text = prepare_tts_input_with_context(text)

    # Generate speech using edge-tts
    try:
        output_file_path = generate_speech(text, voice, response_format, speed)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    # Return the generated audio file
    return send_file(output_file_path, mimetype="audio/mpeg", as_attachment=True, download_name="speech.mp3")

print(f" Edge TTS (Free Azure TTS) Replacement for OpenAI's TTS API")
print(f" ")
print(f" * Serving OpenAI Edge TTS")
print(f" * Server running on http://localhost:{PORT}")
print(f" * TTS Endpoint: http://localhost:{PORT}/v1/audio/speech")
print(f" ")

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', PORT), app)
    http_server.serve_forever()
