#!/usr/bin/env python3
"""
Web server for Imagai - AI Image Generation Web Interface
Provides REST API endpoints to execute CLI commands from the web interface.
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import base64
import mimetypes

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from imagai.config import settings
from imagai.core import generate_image_core
from imagai.models import ImageGenerationRequest

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = Path('generated_images')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


@app.route('/')
def index():
    """Serve the main web interface"""
    try:
        with open('web_interface.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Web interface file not found. Please ensure web_interface.html exists.", 404


@app.route('/api/engines')
def get_engines():
    """Get available engines from configuration"""
    try:
        engines = []
        for engine_name, engine_config in settings.engines.items():
            engines.append({
                'name': engine_name,
                'model': engine_config.model,
                'base_url': engine_config.base_url
            })
        
        return jsonify({
            'success': True,
            'engines': engines,
            'default_engine': settings.default_engine
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/generate', methods=['POST'])
def generate_image():
    """Generate image using the core functionality"""
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400
        
        # Extract parameters from request
        prompt = data['prompt']
        engine = data.get('engine') or settings.default_engine
        
        # Input image for image-to-image generation
        input_image = data.get('input_image')
        
        if not engine:
            return jsonify({
                'success': False,
                'error': 'No engine specified and no default engine configured'
            }), 400
        
        if engine not in settings.engines:
            return jsonify({
                'success': False,
                'error': f'Engine "{engine}" is not configured'
            }), 400
        
        # Build extra parameters
        extra_params = {}
        param_mapping = {
            'negative_prompt': 'negative_prompt',
            'seed': 'seed',
            'strength': 'strength',
            'output_format': 'output_format',
            'aspect_ratio': 'aspect_ratio',
            'mode': 'mode'
        }
        
        for key, param_key in param_mapping.items():
            if key in data and data[key]:
                if key in ['seed'] and data[key]:
                    extra_params[param_key] = int(data[key])
                elif key in ['strength'] and data[key]:
                    extra_params[param_key] = float(data[key])
                else:
                    extra_params[param_key] = data[key]
        
        # Create request object
        image_request = ImageGenerationRequest(
            prompt=prompt,
            engine=engine,
            output_filename=data.get('output'),
            n=int(data.get('n', 1)),
            size=data.get('size', '1024x1024'),
            quality=data.get('quality', 'standard'),
            style=data.get('style', 'vivid'),
            response_format=data.get('response_format', 'b64_json'),
            extra_params=extra_params,
            verbose=data.get('verbose', False),
            auto_filename=data.get('auto_filename', False),
            random_filename=data.get('random_filename', False)
        )
        
        # Process input image if provided
        if input_image:
            # Convert base64 data URL to bytes
            if input_image.startswith('data:image/'):
                # Extract base64 data from data URL
                header, encoded = input_image.split(',', 1)
                image_data = base64.b64decode(encoded)
                # Add to extra_params for the core function
                extra_params['input_image'] = image_data
            else:
                # Assume it's already base64 encoded
                image_data = base64.b64decode(input_image)
                extra_params['input_image'] = image_data
        
        # Update the request object with processed input image
        image_request = ImageGenerationRequest(
            prompt=prompt,
            engine=engine,
            output_filename=data.get('output'),
            n=int(data.get('n', 1)),
            size=data.get('size', '1024x1024'),
            quality=data.get('quality', 'standard'),
            style=data.get('style', 'vivid'),
            response_format=data.get('response_format', 'b64_json'),
            extra_params=extra_params,
            verbose=data.get('verbose', False),
            auto_filename=data.get('auto_filename', False),
            random_filename=data.get('random_filename', False)
        )
        
        # Generate image using core functionality
        async def _generate():
            return await generate_image_core(image_request)
        
        results = asyncio.run(_generate())
        
        # Process results
        response_data = {
            'success': True,
            'results': [],
            'command': f'Generated using engine: {engine}'
        }
        
        for i, result in enumerate(results):
            result_data = {
                'index': i + 1,
                'success': not result.error,
                'error': result.error
            }
            
            if not result.error:
                if result.saved_path:
                    result_data['saved_path'] = result.saved_path
                    # Try to encode image as base64 for preview
                    try:
                        if os.path.exists(result.saved_path):
                            with open(result.saved_path, 'rb') as img_file:
                                img_data = img_file.read()
                                img_b64 = base64.b64encode(img_data).decode('utf-8')
                                mime_type = mimetypes.guess_type(result.saved_path)[0] or 'image/png'
                                result_data['image_data'] = f'data:{mime_type};base64,{img_b64}'
                    except Exception as e:
                        result_data['preview_error'] = str(e)
                
                if result.image_url:
                    result_data['image_url'] = result.image_url
                
                if result.image_b64_json:
                    result_data['image_b64_json'] = result.image_b64_json
            
            response_data['results'].append(result_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/api/generate-cli', methods=['POST'])
def generate_image_cli():
    """Generate image using CLI command execution (alternative method)"""
    try:
        data = request.get_json()
        
        if not data or 'command' not in data:
            return jsonify({
                'success': False,
                'error': 'Command is required'
            }), 400
        
        command = data['command']
        
        # Security: Only allow imagai commands
        if not command.strip().startswith(('rye run imagai', 'python -m imagai', 'imagai')):
            return jsonify({
                'success': False,
                'error': 'Only imagai commands are allowed'
            }), 400
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.getcwd()
            )
            
            response_data = {
                'success': result.returncode == 0,
                'command': command,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            # Try to find generated images
            if result.returncode == 0:
                generated_images = []
                # Look for recently created images in the generated_images directory
                if UPLOAD_FOLDER.exists():
                    for img_file in UPLOAD_FOLDER.glob('*.png'):
                        # Check if file was created recently (within last 5 minutes)
                        if (datetime.now().timestamp() - img_file.stat().st_mtime) < 300:
                            try:
                                with open(img_file, 'rb') as f:
                                    img_data = f.read()
                                    img_b64 = base64.b64encode(img_data).decode('utf-8')
                                    generated_images.append({
                                        'filename': img_file.name,
                                        'path': str(img_file),
                                        'data': f'data:image/png;base64,{img_b64}'
                                    })
                            except Exception as e:
                                print(f"Error reading image {img_file}: {e}")
                
                response_data['generated_images'] = generated_images
            
            return jsonify(response_data)
            
        except subprocess.TimeoutExpired:
            return jsonify({
                'success': False,
                'error': 'Command timed out after 5 minutes'
            }), 408
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Command execution failed: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/images/<filename>')
def serve_image(filename):
    """Serve generated images"""
    try:
        filename = secure_filename(filename)
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404


@app.route('/api/images')
def list_images():
    """List all generated images"""
    try:
        images = []
        if UPLOAD_FOLDER.exists():
            for img_file in UPLOAD_FOLDER.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                    stat = img_file.stat()
                    images.append({
                        'filename': img_file.name,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'url': f'/api/images/{img_file.name}'
                    })
        
        # Sort by creation time, newest first
        images.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'success': True,
            'images': images
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    print("🎨 Starting Imagai Web Server...")
    print(f"📁 Generated images will be saved to: {UPLOAD_FOLDER.absolute()}")
    print(f"🔧 Available engines: {list(settings.engines.keys())}")
    print(f"🌐 Web interface will be available at: http://localhost:5000")
    print("\n" + "="*50)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )