"""
SCT Fusion Query - Web Interface
---------------------------------
Flask-based UI for the Semantic-Condition Transformation system
with Conditional RAG integration.
"""

from flask import Flask, jsonify, request, render_template
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project paths to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'System_KG_Implementation'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RAG_Implementation'))

# Import SCT fusion query module
try:
    from SCT_fusion_query import fused_reasoning, generate_followups
    logger.info("✓ SCT_fusion_query module imported successfully")
except Exception as e:
    logger.error(f"✗ Failed to import SCT_fusion_query: {e}")
    logger.error("Make sure all dependencies are installed and paths are correct")
    sys.exit(1)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sct-fusion-query-2025'

# Store conversation history (in production, use Redis or database)
conversation_history = []


@app.route('/')
def index():
    """Render main chat interface"""
    return render_template('index.html')


@app.route('/get-response', methods=['POST'])
def get_response():
    """
    Process user query and return SCT fusion answer.
    
    Request JSON:
        {
            "query": str,
            "get_followup": bool (optional)
        }
    
    Response JSON:
        {
            "answer": str,
            "sct_context": str,
            "rag_context": str,
            "is_binning": bool,
            "followup_questions": list (if requested)
        }
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        get_followup = data.get('get_followup', False)
        
        if not query:
            return jsonify({
                'error': 'Empty query',
                'answer': 'Please enter a valid question.'
            }), 400
        
        logger.info(f"Processing query: {query}")
        
        # Get answer from SCT fusion system
        answer, sct_context, rag_context, is_binning = fused_reasoning(query)
        
        # Store in conversation history
        conversation_history.append({
            'query': query,
            'answer': answer,
            'is_binning': is_binning
        })
        
        response_data = {
            'answer': answer,
            'sct_context': sct_context,
            'rag_context': rag_context,
            'is_binning': is_binning,
            'query': query
        }
        
        # Generate follow-up questions if requested
        if get_followup:
            try:
                followup_text = generate_followups(query, answer)
                # Parse numbered list into array
                followup_list = []
                for line in followup_text.split('\n'):
                    line = line.strip()
                    # Match lines like "1. question" or "1) question"
                    if line and any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, 10)):
                        # Remove the number prefix
                        question = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                        if question:
                            followup_list.append(question)
                response_data['followup_questions'] = followup_list
            except Exception as e:
                logger.warning(f"Failed to generate follow-up questions: {e}")
                response_data['followup_questions'] = []
        
        logger.info(f"✓ Response generated (Binning: {is_binning})")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'answer': f'An error occurred while processing your question: {str(e)}'
        }), 500


@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    logger.info("Conversation history cleared")
    return jsonify({'status': 'success', 'message': 'Conversation reset'})


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'SCT Fusion Query UI',
        'conversations': len(conversation_history)
    })


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Starting SCT Fusion Query Web Interface")
    logger.info("=" * 60)
    logger.info("Access the UI at: http://localhost:5000")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
