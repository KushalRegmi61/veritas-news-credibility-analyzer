import streamlit as st
import requests
import pandas as pd
import re
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os



load_dotenv()
# Configuration
API_URL = os.getenv('API_URL', "http://127.0.0.1:8000/predict")
MAX_TEXT_LENGTH = 20000
MIN_TEXT_LENGTH = 50

st.set_page_config(
    page_title="Veritas: News Credibility Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Enhanced Helper Functions
def calculate_text_metrics(text: str) -> Dict:
    """Calculate comprehensive metadata features from text"""
    sentences = len(re.findall(r'[.!?]+', text))
    words = text.split()
    
    return {
        'text_length': len(text),
        'word_count': len(words),
        'sentence_count': sentences,
        'avg_sentence_length': len(words) / max(sentences, 1),
        'uppercase_words_count': sum(1 for w in words if w.isupper()),
        'questions_mark_count': text.count('?'),
        'exclamations_mark_count': text.count('!'),
        'numbers_count': len(re.findall(r'\d+', text)),
        'urls_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'email_count': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'readability_score': calculate_readability_score(text)
    }

def calculate_readability_score(text: str) -> float:
    """Simple readability score based on sentence and word length"""
    words = text.split()
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences == 0:
        return 0
    
    avg_sentence_length = len(words) / sentences
    # Simplified readability score (lower is easier to read)
    return min(100, avg_sentence_length * 2)

def get_enhanced_risk_indicators(metrics: Dict) -> List[Dict]:
    """Enhanced risk indicators with severity levels"""
    indicators = []
    
    # High severity indicators
    if metrics['uppercase_words_count'] > 10:
        indicators.append({
            'message': f"Excessive use of capital letters ({metrics['uppercase_words_count']} words in caps)",
            'severity': 'high'
        })
    elif metrics['uppercase_words_count'] > 5:
        indicators.append({
            'message': f"High use of capital letters ({metrics['uppercase_words_count']} words in caps)",
            'severity': 'medium'
        })
    
    if metrics['exclamations_mark_count'] > 5:
        indicators.append({
            'message': f"Very high emotional language ({metrics['exclamations_mark_count']} exclamation marks)",
            'severity': 'high'
        })
    elif metrics['exclamations_mark_count'] > 3:
        indicators.append({
            'message': f"High emotional language ({metrics['exclamations_mark_count']} exclamation marks)",
            'severity': 'medium'
        })
    
    if metrics['questions_mark_count'] > 5:
        indicators.append({
            'message': f"Excessive rhetorical questions ({metrics['questions_mark_count']} question marks)",
            'severity': 'medium'
        })
    
    if metrics['text_length'] < MIN_TEXT_LENGTH:
        indicators.append({
            'message': f"Unusually short content ({metrics['text_length']} characters)",
            'severity': 'medium'
        })
    
    if metrics['readability_score'] > 50:
        indicators.append({
            'message': f"Complex sentence structure (readability score: {metrics['readability_score']:.1f})",
            'severity': 'low'
        })
    
    if metrics['urls_count'] > 3:
        indicators.append({
            'message': f"Multiple external links ({metrics['urls_count']} URLs found)",
            'severity': 'low'
        })
    
    return indicators

def get_confidence_level(confidence: float) -> tuple:
    """Return confidence level description and color based on 0.45 threshold"""
    if confidence < 0.15:
        return "Highly Reliable", "success"
    elif confidence < 0.30:
        return "Likely Reliable", "success" 
    elif confidence < 0.45:
        return "Potentially Concerning", "warning"
    elif confidence < 0.65:
        return "Likely Misinformation", "error"
    else:
        return "High Risk of Misinformation", "error"



def is_lorem_ipsum(text: str) -> bool:
    """
    Detect if text contains Lorem Ipsum placeholder content.
    """
    lorem_words = [
        'lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 
        'adipiscing', 'elit', 'sed', 'do', 'eiusmod', 'tempor',
        'incididunt', 'labore', 'dolore', 'magna', 'aliqua', 'enim',
        'ad', 'minim', 'veniam', 'quis', 'nostrud', 'exercitation',
        'ullamco', 'laboris', 'nisi', 'aliquip', 'ex', 'ea', 'commodo',
        'consequat', 'duis', 'aute', 'irure', 'reprehenderit', 'voluptate',
        'velit', 'esse', 'cillum', 'fugiat', 'nulla', 'pariatur',
        'excepteur', 'sint', 'occaecat', 'cupidatat', 'non', 'proident',
        'sunt', 'in', 'culpa', 'qui', 'officia', 'deserunt', 'mollit',
        'anim', 'id', 'est', 'laborum'
    ]
    
    # Clean and split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    if len(words) == 0:
        return False
    
    # Count Lorem Ipsum words
    lorem_count = sum(1 for word in words if word in lorem_words)
    word_count = sum(1 for word in words if word not in lorem_words)
    
    
    # If more than 20% of words are Lorem Ipsum words, it's likely Lorem Ipsum
    return (lorem_count / word_count) > 0.2

def validate_input(text: str) -> tuple:
    """Enhanced input validation - returns (is_valid, error_message, error_type)"""
    validation_result = validate_news_input(text)
    return validation_result['valid'], validation_result['error'], validation_result.get('type', 'unknown')

def validate_news_input(text: str) -> Dict[str, any]:
    """
    Validate news input text for various issues.
    
    Returns:
        Dict with 'valid' (bool), 'error' (str), and 'type' (str) keys
    """
    
    # Check if text is empty or only whitespace
    if not text or text.strip() == "":
        return {
            'valid': False,
            'error': "⚠️ Please enter some text to analyze.",
            'type': 'empty'
        }
    
    # Check minimum length
    if len(text.strip()) < MIN_TEXT_LENGTH:
        return {
            'valid': False,
            'error': f"⚠️ Please enter at least {MIN_TEXT_LENGTH} characters for meaningful analysis.",
            'type': 'too_short'
        }
    
    # Check maximum length
    if len(text) > MAX_TEXT_LENGTH:
        return {
            'valid': False,
            'error': f"⚠️ Text too long. Please enter no more than {MAX_TEXT_LENGTH} characters.",
            'type': 'too_long'
        }
    
    # Check for Lorem Ipsum
    if is_lorem_ipsum(text):
        return {
            'valid': False,
            'error': "🚫 Lorem Ipsum placeholder text detected. Please enter actual news content to analyze.",
            'type': 'lorem_ipsum'
        }
    
    # Check for excessive repetition (same character repeated)
    if re.search(r'(.)\1{10,}', text):
        return {
            'valid': False,
            'error': "⚠️ Excessive character repetition detected. Please enter readable text.",
            'type': 'repetitive'
        }
    
    # Check word count (should have reasonable number of words)
    words = re.findall(r'\b\w+\b', text)
    if len(words) < 10:
        return {
            'valid': False,
            'error': "⚠️ Please enter at least 10 words for analysis.",
            'type': 'too_few_words'
        }
    
    # Check for excessive special characters
    special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\-\'"()]', text)) / len(text)
    if special_char_ratio > 0.3:
        return {
            'valid': False,
            'error': "⚠️ Too many special characters detected. Please enter readable news text.",
            'type': 'special_chars'
        }
    
    # All validations passed
    return {
        'valid': True,
        'error': None,
        'type': 'valid'
    }

def display_validation_error(error_info: Dict[str, any]) -> None:
    """
    Display validation error in Streamlit with appropriate styling.
    """
    if error_info['type'] == 'lorem_ipsum':
        st.error(error_info['error'])
        st.info(" **Tip**: Lorem Ipsum is placeholder text used in design. Please paste actual news content for analysis.")
    elif error_info['type'] == 'empty':
        st.warning(error_info['error'])
    elif error_info['type'] in ['too_short', 'too_few_words']:
        st.warning(error_info['error'])
        st.info(" **Tip**: News articles are typically longer. Try pasting a complete news article or paragraph.")
    elif error_info['type'] == 'too_long':
        st.error(error_info['error'])
        st.info(f" **Tip**: Try analyzing shorter sections or summarize the content to under {MAX_TEXT_LENGTH} characters.")
    else:
        st.error(error_info['error'])

def make_api_request(payload: Dict, max_retries: int = 3) -> Optional[Dict]:
    """Make API request with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
                return None
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to connect to the API after {max_retries} attempts: {e}")
                return None
            else:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None

def save_analysis_to_history(text: str, result: Dict, metrics: Dict):
    """Save analysis to session history"""
    analysis = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'text_preview': text[:100] + "..." if len(text) > 100 else text,
        'result': result,
        'metrics': metrics,
        'full_text': text
    }
    st.session_state.analysis_history.append(analysis)
    # Keep only last 10 analyses
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[-10:]



def create_metrics_visualization(metrics: Dict) -> go.Figure:
    """
    Create an interactive radar chart visualization to analyze writing patterns 
    of the input text.

    Parameters:
    - metrics: Dict
        Dictionary containing text-related metrics:
            - text_length: int
            - readability_score: float
            - exclamations_mark_count: int
            - questions_mark_count: int
            - uppercase_words_count: int

    Returns:
    - Plotly radar chart figure showing scaled metric values from 0 to 100.
    """

    # Define metric labels with intuitive names
    categories = [
        'Text Length (normalized)',
        'Readability (Flesch score)',
        'Emotional Tone (! marks)',
        'Interrogative Tone (? marks)',
        'Emphasis (CAPS usage)'
    ]

    # Normalize each metric to a 0–100 scale for comparison
    values = [
        min(100, metrics['text_length'] / 20),                    # Long = higher
        min(100, metrics['readability_score']),                   # Higher = easier to read
        min(100, metrics['exclamations_mark_count'] * 10),        # More ! = higher emotional tone
        min(100, metrics['questions_mark_count'] * 20),           # More ? = more questioning
        min(100, metrics['uppercase_words_count'] * 5)            # CAPS = more emphasis
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Writing Pattern Score',
        line=dict(color='indigo'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title={
            'text': "Writing Pattern Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        polar=dict(
            bgcolor='#f9f9f9',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                title="Score"
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
        height=400
    )

    return fig


import streamlit as st
import plotly.graph_objects as go
from typing import Dict

def create_metrics_visualization(metrics: Dict) -> go.Figure:
    """
    Create an interactive radar chart visualization to analyze writing patterns 
    of the input text.

    Parameters:
    - metrics: Dict
        Dictionary containing text-related metrics:
            - text_length: int
            - readability_score: float
            - exclamations_mark_count: int
            - questions_mark_count: int
            - uppercase_words_count: int

    Returns:
    - Plotly radar chart figure showing scaled metric values from 0 to 100.
    """

    categories = [
        'Text Length (normalized)',
        'Readability (Flesch score)',
        'Emotional Tone (! marks)',
        'Interrogative Tone (? marks)',
        'Emphasis (CAPS usage)'
    ]

    values = [
        min(100, metrics['text_length'] / 20),
        min(100, metrics['readability_score']),
        min(100, metrics['exclamations_mark_count'] * 10),
        min(100, metrics['questions_mark_count'] * 20),
        min(100, metrics['uppercase_words_count'] * 5)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Writing Pattern Score',
        line=dict(color='indigo'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title={
            'text': "Writing Pattern Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        polar=dict(
            bgcolor='#f9f9f9',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=100, b=40),
        height=430
    )

    return fig

def display_metrics_with_summary(metrics: Dict):
    """
    Display the radar chart with a left-side one-line summary.
    """
    fig = create_metrics_visualization(metrics)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("### Summary")
        st.write(
        "This radar chart highlights key writing traits length, readability, emotional tone, questioning, "
        "and emphasis to show how writing style may affect reader trust and content perception."
    )


    with right_col:
        st.plotly_chart(fig, use_container_width=True)


def categorize_shap_features(feature_names: List[str], shap_values: List[float]) -> Dict:
    """Categorize SHAP features into different types for better presentation"""
    categories = {
        'text_features': [],
        'metadata_features': [],
        'positive_impact': [],
        'negative_impact': []
    }
    
    metadata_keywords = ['text_length', 'exclamations_mark_count', 'questions_mark_count', 'uppercase_words_count']
    
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
        # Categorize by feature type
        if any(keyword in feature for keyword in metadata_keywords):
            categories['metadata_features'].append({'feature': feature, 'value': shap_val, 'index': i})
        else:
            categories['text_features'].append({'feature': feature, 'value': shap_val, 'index': i})
        
        # Categorize by impact direction
        if shap_val > 0:
            categories['positive_impact'].append({'feature': feature, 'value': shap_val, 'index': i})
        else:
            categories['negative_impact'].append({'feature': feature, 'value': shap_val, 'index': i})
    
    return categories

# Enhanced UI Components
def display_enhanced_sidebar():
    """Enhanced sidebar with additional features"""
    with st.sidebar:
        
        st.markdown("### About")
        st.write(
            "Advanced ML system that predicts whether news articles are fake or real "
            "with explainable AI insights and risk assessment."
        )
        # st.markdown('---')
        # Model info
        with st.expander(" Model Information"):
            st.write("Algorithm: XGBoost Classifier")
            st.write("Features: text and metadata features")
            st.write("Training Data: 30,000+ verified articles")
            st.write("Performance: 98.48% F1-Score")
            st.write("Last Updated: July 2025")
        
        st.markdown('---')
        # Analysis history
        st.markdown("###  Analysis History")
        if st.session_state.analysis_history:
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                with st.expander(f"Analysis {len(st.session_state.analysis_history) - i}"):
                    st.write(f"**Time:** {analysis['timestamp']}")
                    st.write(f"**Preview:** {analysis['text_preview']}")
                    
                    # Add predicted class and probabilities
                    result = analysis['result']
                    predicted_class = result.get('predicted_category', 'Unknown')
                    confidence = result.get('confidence', 0)
                    
                    # Show prediction with color coding
                    if predicted_class == 'FAKE NEWS':
                        st.error(f"**Prediction:** {predicted_class}")
                    else:
                        st.success(f"**Prediction:** {predicted_class}")
                    
                    st.write(f"**Confidence:** {confidence:.1%}")
                    
                    # Class probabilities
                    class_probs = result.get('class_probabilities', {})
                    if class_probs:
                        st.write("**Probabilities:**")
                        for category, prob in class_probs.items():
                            st.write(f"  - {category}: {prob:.1%}")
        else:
            st.info("No analyses yet. Start by analyzing an article!")
        
        # Export option
        if st.session_state.analysis_history:
            st.markdown("### Export")
            if st.button("Export History"):
                export_data = json.dumps(st.session_state.analysis_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"veritas_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
def display_shap_explanation(explanation: Dict):
    """Display SHAP explanation in a professional manner"""
    st.markdown("---")
    
    st.markdown("#### Model Decision Factors")
    st.markdown("""
    This section shows the key factors that influenced the model's decision, 
    ranked by importance. Positive values push toward "Fake News" classification, 
    while negative values push toward "Real News" classification.
    """)
    
    if explanation:
        # Create DataFrame for better display
        shap_df = pd.DataFrame({
            'Feature': explanation['feature_names'],
            'SHAP Value': explanation['shap_values'],
            'Feature Value': explanation['feature_values'],
            'Impact': ['Positive' if val > 0 else 'Negative' for val in explanation['shap_values']]
        })
        
        # Categorize features for better presentation
        categories = categorize_shap_features(explanation['feature_names'], explanation['shap_values'])
        
        # Create two columns for positive and negative impacts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Factors Supporting 'Fake News' Classification:**")
            if categories['positive_impact']:
                for item in sorted(categories['positive_impact'], key=lambda x: x['value'], reverse=True)[:5]:
                    st.write(f"• {item['feature']}: {item['value']:.4f}")
            else:
                st.write("No strong positive indicators found")
        
        with col2:
            st.markdown("**Factors Supporting 'Real News' Classification:**")
            if categories['negative_impact']:
                for item in sorted(categories['negative_impact'], key=lambda x: abs(x['value']), reverse=True)[:5]:
                    st.write(f"• {item['feature']}: {item['value']:.4f}")
            else:
                st.write("No strong negative indicators found")
        
        
        # Create SHAP visualization
        fig_shap = go.Figure()
        
        colors = ['red' if val > 0 else 'green' for val in shap_df['SHAP Value']]
        
        fig_shap.add_trace(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{val:.4f}' for val in shap_df['SHAP Value']],
            textposition='auto',
            name='SHAP Values'
        ))
        
        fig_shap.update_layout(
            title='Feature Impact on Prediction',
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='Features',
            height=max(400, len(shap_df) * 25),
            showlegend=False
        )
        
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Display feature details in expandable section
        with st.expander("Detailed Feature Analysis"):
            st.markdown('---')
            # interpretation guide
            st.markdown("""
            **Interpretation Guide:**
            - **SHAP Value > 0**: Feature pushes prediction toward "Fake News"
            - **SHAP Value < 0**: Feature pushes prediction toward "Real News"
            - **Magnitude**: Larger absolute values indicate stronger influence
            - **Base Value**: Starting point before considering any features
            - **Feature Value**: The actual value of the feature in the analyzed text
            """)
            
            st.markdown(f"**Base Value:** {explanation['base_value']:.4f}  <span style='color:gray'>(The model's starting prediction before considering any features)</span>", unsafe_allow_html=True)
            st.markdown('---')
            
            
            
            # Separate metadata and text features
            metadata_df = shap_df[shap_df['Feature'].str.contains('text_length|exclamations_mark_count|questions_mark_count|uppercase_words_count', na=False)]
            text_df = shap_df[~shap_df['Feature'].str.contains('text_length|exclamations_mark_count|questions_mark_count|uppercase_words_count', na=False)]
            
            if not metadata_df.empty:
                st.markdown("**Metadata Features:**")
                st.dataframe(
                    metadata_df.style.format({
                        'SHAP Value': '{:.4f}',
                        'Feature Value': '{:.4f}'
                    }),
                    use_container_width=True
                )
            
            if not text_df.empty:
                st.markdown("**Text Features (Top Words/Phrases):**")
                st.dataframe(
                    text_df.style.format({
                        'SHAP Value': '{:.4f}',
                        'Feature Value': '{:.4f}'
                    }),
                    use_container_width=True
                )
            
        
        # Summary insights
        st.markdown("#### Key Insights")
        total_positive = sum(val for val in explanation['shap_values'] if val > 0)
        total_negative = sum(val for val in explanation['shap_values'] if val < 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive Impact Sum", f"{total_positive:.4f}")
        with col2:
            st.metric("Negative Impact Sum", f"{total_negative:.4f}")
        with col3:
            st.metric("Net Impact", f"{total_positive + total_negative:.4f}")
        
        # Professional interpretation
        if abs(total_positive) > abs(total_negative):
            st.info("**Model Reasoning:** The positive features (supporting fake news classification) outweigh the negative features in this analysis.")
        else:
            st.info("**Model Reasoning:** The negative features (supporting real news classification) outweigh the positive features in this analysis.")


def render_risk_meter(fake_conf: float, text_description: str | None, threshold: float = 45.0):
    """
    Render a compact, professional Risk Meter showing fake news probability.

    Parameters:
    - fake_conf: float : Model's predicted probability (0 to 1) that input is fake news.
    - threshold: float : Threshold (%) above which content is flagged as risky.
    """

    left_col, right_col = st.columns([2, 1])

    # Left side: Explanation
    with left_col:
        st.subheader("Assessment Summary")
        st.markdown(f"""
        **Fake News Probability:** `{fake_conf * 100:.1f}%`  
        **Threshold:** `{threshold:.0f}%`  
        """)
        # if fake_conf * 100 >= threshold:
        # Use get_confidence_level to determine description and color
        description, color = get_confidence_level(fake_conf)
        if color == "error":
            st.markdown(
            f"<span style='color:crimson; font-weight:600'>{text_description}.</span>", 
            unsafe_allow_html=True
            )
        elif color == "warning":
            st.markdown(
            f"<span style='color:orange; font-weight:600'>{text_description}. </span>", 
            unsafe_allow_html=True
            )
        else:
            st.markdown(
            f"<span style='color:seagreen; font-weight:600'>{text_description}.</span>", 
            unsafe_allow_html=True
            )

    # Right side: Gauge meter
    with right_col:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fake_conf * 100,
            delta={
                'reference': threshold,
                'increasing': {'color': "red"},
                'decreasing': {'color': "green"}
            },
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fake Probability", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "crimson" if fake_conf * 100 >= threshold else "seagreen"},
                'steps': [
                    {'range': [0, 20], 'color': '#d4edda'},
                    {'range': [20, 35], 'color': '#ffeeba'},
                    {'range': [35, threshold], 'color': '#fff3cd'},
                    {'range': [threshold, 65], 'color': '#f8d7da'},
                    {'range': [65, 100], 'color': '#f5c6cb'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': threshold
                }
            }
        ))

        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240)
        st.plotly_chart(fig, use_container_width=True)


def display_enhanced_results(data: Dict, metrics: Dict, risk_indicators: List[Dict]):
    """Display enhanced analysis results with professional formatting"""
    # Extract prediction data
    predicted_category = data.get('predicted_category', 'Unknown')
    confidence = data.get('confidence', 0)
    fake_prob = data['class_probabilities'].get('Fake News', 0)
    real_prob = data['class_probabilities'].get('Real News', 0)
    explanation = data.get('explanation', {})
    
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    # Primary prediction display
    st.markdown("### Classification Result")
    if predicted_category == "Fake News":
        st.error(f"**{predicted_category.upper()}** (Confidence: {confidence:.1%})")
    else:
        st.success(f"**{predicted_category.upper()}** (Confidence: {confidence:.1%})")
    
    # Classification probabilities chart
    st.markdown("### Model Confidence")
    prob_data = {
        'Category': list(data['class_probabilities'].keys()),
        'Probability': [p * 100 for p in data['class_probabilities'].values()]
    }
    prob_df = pd.DataFrame(prob_data)
    
    fig_prob = px.bar(
        prob_df, 
        x='Probability', 
        y='Category',
        title='Classification Confidence by Category',
        color='Probability',
        color_continuous_scale=['#dc3545', '#28a745'],
        text='Probability',
        orientation='h'
    )
    
    fig_prob.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='inside',
        textfont_size=14,
        textfont_color='white'
    )
    
    fig_prob.update_layout(
        height=250,
        showlegend=False,
        xaxis_title="Confidence (%)",
        yaxis_title="",
        font=dict(size=12),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig_prob, use_container_width=True)
    
    st.markdown('---')
    # Risk assessment
    with st.expander('### Risk Assessment'):
        # Determine risk level and description
        if fake_prob < 0.3:
            risk_level = "Low Risk"
            risk_color = "success"
            risk_description = "Content aligns with standard journalistic patterns and shows minimal indicators of misinformation."
        elif fake_prob < 0.6:
            risk_level = "Medium Risk" 
            risk_color = "warning"
            risk_description = "Analysis identified patterns that warrant additional verification before sharing or acting on this information."
        else:
            risk_level = "High Risk"
            risk_color = "error" 
            risk_description = "Multiple indicators suggest this content may contain misleading or fabricated information."
    
        # Display risk assessment
        if risk_color == "success":
            st.success(f"**{risk_level}** - Misinformation probability: {fake_prob:.1%}")
        elif risk_color == "warning":
            st.warning(f"**{risk_level}** - Misinformation probability: {fake_prob:.1%}")
        else:
            st.error(f"**{risk_level}** - Misinformation probability: {fake_prob:.1%}")
        
        # risk meter
        render_risk_meter(fake_prob, risk_description)
        # st.info(risk_description)
    
    st.markdown('---')
    # Classification methodology
    
    st.markdown("#### Classification Methodology")
    st.info(f"""
    **Threshold-based Classification**: Content with ≥45% misinformation probability is classified as fake news.
    
    **Current Analysis**: {fake_prob:.1%} misinformation probability ({'above' if fake_prob >= 0.45 else 'below'} classification threshold)
    
    **Important**: This analysis is based on linguistic patterns and writing style indicators, not content verification. Always cross-reference with authoritative sources.
    """)
    
    # Detailed metrics
    st.markdown("---")
    st.markdown("## Content Analysis Metrics")
    
    # Primary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Word Count", f"{metrics['word_count']:,}")
    with col2:
        st.metric("Sentences", metrics['sentence_count'])
    with col3:
        st.metric("Average Sentence Length", f"{metrics['avg_sentence_length']:.1f}")
    with col4:
        st.metric("Readability Score", f"{metrics['readability_score']:.1f}")
    
    # Style indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Uppercase Words", metrics['uppercase_words_count'])
    with col2:
        st.metric("Exclamation Marks", metrics['exclamations_mark_count'])
    with col3:
        st.metric("Question Marks", metrics['questions_mark_count'])
    with col4:
        st.metric("External URLs", metrics['urls_count'])
    
    # Risk indicators analysis
    if risk_indicators:
        st.markdown("#### Content Pattern Analysis")
        
        # Categorize indicators by severity
        high_indicators = [r for r in risk_indicators if r['severity'] == 'high']
        medium_indicators = [r for r in risk_indicators if r['severity'] == 'medium'] 
        low_indicators = [r for r in risk_indicators if r['severity'] == 'low']
        
        if high_indicators:
            st.error("**High Priority Concerns**")
            for indicator in high_indicators:
                st.write(f"• {indicator['message']}")
        
        if medium_indicators:
            st.warning("**Moderate Concerns**")  
            for indicator in medium_indicators:
                st.write(f"• {indicator['message']}")
        
        if low_indicators:
            st.info("**Minor Observations**")
            for indicator in low_indicators:
                st.write(f"• {indicator['message']}")
    else:
        st.success("#### Pattern Analysis: No Significant Concerns")
        st.write("The content follows standard journalistic writing conventions.")
    
    # Visualization
    st.markdown("---")
    with st.expander('Metric Visualization'):
    
        display_metrics_with_summary(metrics)
        # st.plotly_chart(fig, use_container_width=True)
    
    # SHAP explanation if available
    if explanation:
        display_shap_explanation(explanation)
    
    # Technical details (collapsed by default)
    st.markdown('---')
    with st.expander("Technical Analysis Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Outputs**")
            st.json({
                "classification": predicted_category,
                "confidence": round(confidence, 4),
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "threshold": 0.45
            })
        
        with col2:
            st.markdown("**Content Metrics**")
            technical_metrics = {k: v for k, v in metrics.items() if k != 'text_length'}
            st.json(technical_metrics)
# Main Application
def main():
    display_enhanced_sidebar()
    st.title("Veritas: News Credibility Analyzer")
    # st.markdown("""
    # *Advanced ML system that predicts whether news articles are fake or real with explainable AI insights*
    # """)
    st.info("""
    **Development Notice**: This tool is an independent research project developed to demonstrate advanced NLP techniques for content analysis. 
    While the model delivers consistent results, cross-referencing with multiple sources is recommended for critical decisions.
    """)
      
    st.info("**How it works:** Input news text → ML prediction → SHAP analysis → Risk assessment → Visual explanations")

    # Input methods
    input_method = st.radio(
        "## Choose input method:",
        ["Text Input", "File Upload (Coming Soon)"],
        horizontal=True
    )
    # Short info about the app
  
    
    if input_method == "Text Input":
        text_input = st.text_area(
            label="News Article Text",
            height=300,
            placeholder='Paste your news article here...\n\nExample: "Scientists at MIT announced today a breakthrough discovery in quantum computing that could revolutionize how we process information..."',
            help=f"Enter between {MIN_TEXT_LENGTH} and {MAX_TEXT_LENGTH} characters"
        )
        
        # Character counter
        char_count = len(text_input)
        color = "green" if MIN_TEXT_LENGTH <= char_count <= MAX_TEXT_LENGTH else "red"
        st.markdown(f"<p style='color: {color}'>Characters: {char_count}/{MAX_TEXT_LENGTH}</p>", unsafe_allow_html=True)
        
        # ADD THE EXAMPLE SECTION HERE:
        with st.expander("Try Example Inputs"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Try Lorem Ipsum (Invalid)"):
                    st.text_area(
                        "Example - Lorem Ipsum:",
                        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
                        height=100,
                        disabled=True,
                        key="lorem_example"
                    )
            
            with col2:
                if st.button("Try Valid News Example"):
                    st.text_area(
                        "Example - Valid News:",
                        "The President announced new economic policies today during a press conference at the White House. The policies are expected to impact various sectors of the economy including healthcare, education, and infrastructure development.",
                        height=100,
                        disabled=True,
                        key="valid_example"
                    )
                    
                    
        # Analyze button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_button = st.button("Analyze Content", type="primary")
        with col2:
            if st.button("Save Analysis"):
                st.rerun()
                
                

        if analyze_button:
            # Enhanced validation
            validation_result = validate_news_input(text_input)
            
            if not validation_result['valid']:
                # Display enhanced error messages
                display_validation_error(validation_result)
            else:
                # Calculate metrics
                text_metrics = calculate_text_metrics(text_input)
                risk_indicators = get_enhanced_risk_indicators(text_metrics)
                
                # Make API request
                payload = {"text": text_input}
                
                with st.spinner("Analyzing content patterns. Hold tight, this may take 1-2 minutes (go grab a coffee ☕)!"):
                    response_data = make_api_request(payload)
                
                if response_data:
                    result = response_data.get('response')
                    
                    # Validate response structure
                    if not result or 'predicted_category' not in result:
                        st.error("Invalid API response format")
                        return
                    
                    # Save to history
                    save_analysis_to_history(text_input, result, text_metrics)
                    
                    # Display results
                    display_enhanced_results(result, text_metrics, risk_indicators)
                    
                    # Recommendations with SHAP insights
                    st.markdown('---')
                    st.markdown("## Recommendations")

                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        ### **Verification Steps:**
                        - Cross-reference with 2-3 established news sources
                        - Check author credentials and expertise
                        - Verify publication date and recency
                        - Look for supporting evidence and citations
                        - Check if the story is reported by major outlets
                        """)
                    
                    with col2:
                        st.markdown("""
                        ### **Red Flags to Watch:**
                        - Highly emotional or sensational language
                        - Missing or questionable source citations
                        - Suspicious publication dates
                        - Lack of author information
                        - Claims that seem too good/bad to be true
                        """)
    
    else:
        st.info("File upload feature coming soon! You'll be able to upload PDF, Word, and text files.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        <p>This tool is designed to promote media literacy and critical thinking.<br>
        Always verify important information through multiple reliable sources.</p>
        <p>Questions or feedback? Contact: <a href="https://github.com/kushalregmi61">Kushal Regmi</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()