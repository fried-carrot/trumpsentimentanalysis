import os
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import torch
import warnings
warnings.filterwarnings('ignore')

# Import NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class TwitterSentimentAnalyzer:
    """Enhanced Twitter sentiment analyzer for fiscal policy analysis"""
    
    def __init__(self, load_model=True):
        """Initialize the analyzer with all necessary components"""
        print("Initializing Twitter Fiscal Policy Sentiment Analyzer...")
        
        # Initialize model if requested
        if load_model:
            self._init_sentiment_model()
        else:
            print("Sentiment model loading skipped. Will be loaded when needed.")
            self.model = None
            self.tokenizer = None
        
        # Data storage
        self.datasets = {}
        self.sentiment_results = {}
        self.loaded_splits = set()
        
        # Define date periods
        self._init_date_periods()
        
        # Define reference policies
        self._init_reference_policies()
        
        print("Analyzer ready. Use bulk_upload() to load data files.")
    
    def _init_sentiment_model(self):
        """Initialize the sentiment analysis model"""
        print("Loading sentiment model...")
        try:
            self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            self.model.to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Will attempt to load model again when needed.")
    
    def _init_date_periods(self):
        """Set up the date periods for analysis"""
        self.date_splits = {
            "Split 1": {"name": "Jan 21–27, 2025", "range": (datetime(2025, 1, 21), datetime(2025, 1, 27))},
            "Split 2": {"name": "Jan 28–Feb 3, 2025", "range": (datetime(2025, 1, 28), datetime(2025, 2, 3))},
            "Split 3": {"name": "Feb 4–10, 2025", "range": (datetime(2025, 2, 4), datetime(2025, 2, 10))},
            "Split 4": {"name": "Feb 11–17, 2025", "range": (datetime(2025, 2, 11), datetime(2025, 2, 17))},
            "Split 5": {"name": "Feb 18–24, 2025", "range": (datetime(2025, 2, 18), datetime(2025, 2, 24))},
            "Split 6": {"name": "Feb 25–Mar 3, 2025", "range": (datetime(2025, 2, 25), datetime(2025, 3, 3))},
            "Split 7": {"name": "Mar 4–10, 2025", "range": (datetime(2025, 3, 4), datetime(2025, 3, 10))},
            "Split 8": {"name": "Mar 11–17, 2025", "range": (datetime(2025, 3, 11), datetime(2025, 3, 17))},
            "Split 9": {"name": "Mar 18–24, 2025", "range": (datetime(2025, 3, 18), datetime(2025, 3, 24))}
        }
    
    def _init_reference_policies(self):
        """Set up reference fiscal policies for comparative analysis"""
        self.reference_policies = {
            "Tax Cuts 2017": {
                "description": "Trump's Tax Cuts and Jobs Act of 2017",
                "initial_sentiment": 0.35,
                "short_term_sentiment": 0.42,
                "long_term_sentiment": 0.28,
                "pattern": "rise-then-fall"
            },
            "Tariffs 2018": {
                "description": "China and global tariff implementation",
                "initial_sentiment": -0.15,
                "short_term_sentiment": -0.38,
                "long_term_sentiment": -0.22,
                "pattern": "sharp-fall-then-partial-recovery"
            },
            "Federal Reserve 2019": {
                "description": "Federal Reserve interest rate criticism",
                "initial_sentiment": 0.08,
                "short_term_sentiment": -0.05, 
                "long_term_sentiment": -0.18,
                "pattern": "gradual-decline"
            },
            "COVID Stimulus 2020": {
                "description": "COVID-19 economic stimulus packages",
                "initial_sentiment": 0.56,
                "short_term_sentiment": 0.38,
                "long_term_sentiment": 0.12,
                "pattern": "steady-decline"
            },
            "Infrastructure 2021": {
                "description": "Infrastructure investment proposals",
                "initial_sentiment": 0.22,
                "short_term_sentiment": 0.31,
                "long_term_sentiment": 0.45,
                "pattern": "gradual-rise"
            }
        }
    
    def bulk_upload(self):
        """Upload and process multiple Twitter JSON files at once"""
        print("🚀 BULK UPLOAD MODE")
        print("Automatically loading JSON files from fixed directory.")
 
        # Fixed directory path for macOS
        fixed_directory = "/Users/sharatsakamuri/downloads/pleb sentiment"
 
        # Use glob to get all JSON files with names starting with 'split'
        import glob
        file_paths = glob.glob(os.path.join(fixed_directory, "split*.json"))
 
        if not file_paths:
            print("No JSON files found in the fixed directory:", fixed_directory)
            return None
 
        print(f"Found {len(file_paths)} files in {fixed_directory}")
 
        # Read the selected files and store their contents in a dictionary
        uploaded = {}
        for file_path in file_paths:
            with open(file_path, "rb") as f:
                content = f.read()
            filename = os.path.basename(file_path)
            uploaded[filename] = content

        # Track results
        successful_uploads = []
        failed_uploads = []

        # Process each file
        for filename, content in uploaded.items():
            try:
                # Determine which split this file belongs to
                split_num = self._detect_split_number(filename, content)

                if split_num:
                    # Process the file for this split
                    split_name = f"Split {split_num}"
                    df = self._process_file(split_name, content)

                    if df is not None and len(df) > 0:
                        successful_uploads.append((filename, split_num))
                        print(f"✅ Loaded {len(df)} tweets for {split_name}")
                    else:
                        failed_uploads.append((filename, "No valid data"))
                else:
                    failed_uploads.append((filename, "Could not determine split"))
                    print(f"❌ Failed to determine appropriate split for {filename}")
            except Exception as e:
                failed_uploads.append((filename, str(e)))
                print(f"❌ Error processing {filename}: {str(e)}")

        # Print summary
        print("\n--- UPLOAD SUMMARY ---")
        if successful_uploads:
            print("\n✅ Successfully uploaded:")
            for filename, split_num in successful_uploads:
                split_name = f"Split {split_num}"
                print(f"  • {filename} → {split_name} ({self.date_splits[split_name]['name']})")

        if failed_uploads:
            print("\n❌ Failed uploads:")
            for filename, error in failed_uploads:
                print(f"  • {filename}: {error}")

        # Return results for further use
        return successful_uploads, failed_uploads
    
    def _detect_split_number(self, filename, content):
        """Determine which split a file belongs to based on filename or content"""
        # First, check filename for pattern like split1.json
        split_match = re.search(r'split[\s_-]?(\d)', filename.lower())
        if split_match and split_match.group(1):
            split_num = int(split_match.group(1))
            if 1 <= split_num <= 9:
                print(f"✓ Detected split {split_num} from filename: {filename}")
                return split_num
        
        # If no match in filename, analyze content dates
        print(f"Analyzing content dates in {filename}...")
        try:
            # Parse JSON
            if isinstance(content, bytes):
                data = json.loads(content.decode('utf-8'))
            else:
                data = json.loads(content)
            
            # Extract dates from tweets
            dates = []
            for tweet in data[:20]:  # Sample first 20 tweets
                if 'createdAt' in tweet:
                    try:
                        # Twitter's standard format
                        date = datetime.strptime(tweet['createdAt'], '%a %b %d %H:%M:%S +0000 %Y')
                        dates.append(date)
                    except:
                        # Try alternative formats
                        try:
                            date = pd.to_datetime(tweet['createdAt'])
                            dates.append(date)
                        except:
                            continue
            
            if dates:
                # Find median date
                dates.sort()
                median_date = dates[len(dates) // 2]
                
                # Match to a split period
                for s in range(1, 10):
                    split_key = f"Split {s}"
                    date_range = self.date_splits[split_key]["range"]
                    if date_range[0] <= median_date <= date_range[1]:
                        print(f"✓ Assigned to {split_key} based on dates (median: {median_date.strftime('%Y-%m-%d')})")
                        return s
            
            # If still no match, use first available split
            for s in range(1, 10):
                split_key = f"Split {s}"
                if split_key not in self.loaded_splits:
                    print(f"✓ Assigned to {split_key} (first available)")
                    return s
                    
        except Exception as e:
            print(f"Error analyzing content: {str(e)}")
        
        # If we get here, show interactive prompt to select split
        print(f"Could not automatically determine split for {filename}")
        print("Please select manually:")
        for i in range(1, 10):
            print(f"{i}: Split {i} ({self.date_splits[f'Split {i}']['name']})")
        
        try:
            split_num = int(input("Enter split number (1-9): "))
            if 1 <= split_num <= 9:
                return split_num
        except:
            pass
        
        return None
    
    def _process_file(self, split_name, content):
        """Process a file for a specific split"""
        try:
            # Parse JSON
            if isinstance(content, bytes):
                data = json.loads(content.decode('utf-8'))
            else:
                data = json.loads(content)
            
            # Convert to DataFrame
            df = self._parse_tweets(data, split_name)
            
            # Filter by date range if needed
            if split_name in self.date_splits:
                date_range = self.date_splits[split_name]["range"]
                if 'created_at' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
                        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                    
                    # Filter by date range
                    df = df[(df['created_at'] >= date_range[0]) & (df['created_at'] <= date_range[1])]
            
            # Store in datasets
            self.datasets[split_name] = df
            self.loaded_splits.add(split_name)
            
            return df
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return None
    
    def _parse_tweets(self, tweets_data, split_name):
        """Parse Twitter JSON data into a structured DataFrame"""
        parsed_data = []
        
        for tweet in tweets_data:
            # Extract basic tweet info
            tweet_info = {
                'tweet_id': tweet.get('id', ''),
                'text': tweet.get('text', '') or tweet.get('fullText', ''),
                'created_at': tweet.get('createdAt', ''),
                'retweet_count': tweet.get('retweetCount', 0),
                'reply_count': tweet.get('replyCount', 0),
                'like_count': tweet.get('likeCount', 0),
                'view_count': tweet.get('viewCount', 0),
                'is_reply': tweet.get('isReply', False),
                'is_retweet': tweet.get('isRetweet', False),
                'is_quote': tweet.get('isQuote', False),
                'split': split_name
            }
            
            # Extract author info if available
            if 'author' in tweet and tweet['author'] is not None:
                tweet_info.update({
                    'author_id': tweet['author'].get('id', ''),
                    'author_name': tweet['author'].get('name', ''),
                    'author_username': tweet['author'].get('userName', ''),
                    'author_followers': tweet['author'].get('followers', 0),
                    'author_verified': tweet['author'].get('isVerified', False) or 
                                      tweet['author'].get('isBlueVerified', False)
                })
            
            # Convert created_at to datetime
            if tweet_info['created_at']:
                try:
                    # Parse Twitter's datetime format
                    dt = datetime.strptime(tweet_info['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    tweet_info['created_at'] = dt
                except:
                    try:
                        dt = pd.to_datetime(tweet_info['created_at'])
                        tweet_info['created_at'] = dt
                    except:
                        pass
            
            parsed_data.append(tweet_info)
        
        # Create DataFrame
        df = pd.DataFrame(parsed_data)
        
        # Add clean text for analysis
        if 'text' in df.columns:
            df['clean_text'] = df['text'].apply(self._clean_text)
        
        return df
    
    def _clean_text(self, text):
        """Clean and preprocess tweet text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbol but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_all_splits(self):
        """Run sentiment analysis on all loaded data splits"""
        if not self.loaded_splits:
            print("No data loaded. Please upload data first using bulk_upload().")
            return None
        
        # Ensure model is loaded
        if self.model is None:
            self._init_sentiment_model()
            
        if self.model is None:
            print("Failed to load sentiment model. Cannot perform analysis.")
            return None
        
        print(f"Running sentiment analysis on {len(self.loaded_splits)} data splits...")
        
        results = {}
        for split_name in sorted(self.loaded_splits):
            print(f"\nProcessing {split_name}...")
            df, stats = self.analyze_split(split_name)
            results[split_name] = stats
        
        print("\nAnalysis complete for all splits!")
        return results
    
    def analyze_split(self, split_name):
        """Analyze sentiment for a specific data split"""
        if split_name not in self.datasets:
            print(f"No data found for {split_name}. Please load data first.")
            return None, None
        
        # Ensure model is loaded
        if self.model is None:
            self._init_sentiment_model()
            
        if self.model is None:
            print("Failed to load sentiment model. Cannot perform analysis.")
            return None, None
        
        df = self.datasets[split_name]
        
        # Skip if already analyzed
        if split_name in self.sentiment_results:
            print(f"Using cached results for {split_name}")
            return self.sentiment_results[split_name], self._calculate_stats(self.sentiment_results[split_name])
        
        print(f"Analyzing sentiment for {len(df)} tweets in {split_name}...")
        
        # Process in batches
        texts = df['clean_text'].tolist()
        sentiment_results = []
        
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:min(i+batch_size, len(texts))]
            batch_results = self._analyze_sentiment_batch(batch_texts)
            sentiment_results.extend(batch_results)
            
            # Show progress
            if (i+batch_size) % 160 == 0 or i+batch_size >= len(texts):
                print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)} tweets...")
        
        # Add sentiment results to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        if len(sentiment_df) < len(df):
            # Pad with empty results if needed
            sentiment_df = pd.concat([
                sentiment_df, 
                pd.DataFrame([{
                    'negative_prob': 0, 
                    'neutral_prob': 0, 
                    'positive_prob': 0, 
                    'sentiment_score': 0, 
                    'sentiment_label': 'neutral'
                }] * (len(df) - len(sentiment_df)))
            ])
        
        # Combine with original data
        result_df = pd.concat([
            df.reset_index(drop=True), 
            sentiment_df.reset_index(drop=True)
        ], axis=1)
        
        # Store results
        self.sentiment_results[split_name] = result_df
        
        # Calculate statistics
        stats = self._calculate_stats(result_df)
        
        # Print summary
        print(f"Analysis complete for {split_name}:")
        print(f"Average Sentiment: {stats['avg_sentiment_score']:.4f}")
        print(f"Positive: {stats['positive_pct']:.1f}%, Neutral: {stats['neutral_pct']:.1f}%, Negative: {stats['negative_pct']:.1f}%")
        
        return result_df, stats
    
    def _analyze_sentiment_batch(self, texts, batch_size=16):
        """Process a batch of texts with the sentiment model"""
        results = []
        
        # Skip empty texts
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            return results
        
        # Tokenize
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process results
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        batch_results = probabilities.cpu().numpy()
        
        for prob in batch_results:
            # Calculate sentiment score and label
            sentiment_score = prob[2] - prob[0]  # positive - negative
            sentiment_label = ["negative", "neutral", "positive"][prob.argmax()]
            
            results.append({
                'negative_prob': float(prob[0]),
                'neutral_prob': float(prob[1]),
                'positive_prob': float(prob[2]),
                'sentiment_score': float(sentiment_score),
                'sentiment_label': sentiment_label
            })
        
        return results
    
    def _calculate_stats(self, df):
        """Calculate sentiment statistics for a dataset"""
        stats = {
            'total_tweets': len(df),
            'avg_sentiment_score': df['sentiment_score'].mean(),
            'positive_pct': (df['sentiment_label'] == 'positive').mean() * 100,
            'neutral_pct': (df['sentiment_label'] == 'neutral').mean() * 100,
            'negative_pct': (df['sentiment_label'] == 'negative').mean() * 100
        }
        
        # Calculate influencer impact if possible
        if 'author_followers' in df.columns and 'sentiment_score' in df.columns:
            total_followers = df['author_followers'].sum()
            if total_followers > 0:
                stats['influencer_impact'] = (df['sentiment_score'] * df['author_followers']).sum() / total_followers
            else:
                stats['influencer_impact'] = df['sentiment_score'].mean()
        else:
            stats['influencer_impact'] = df['sentiment_score'].mean()
        
        return stats
    
    def get_trend_data(self):
        """Get sentiment trend data across all analyzed splits"""
        if not self.sentiment_results:
            print("No analyzed data available. Run analyze_all_splits() first.")
            return None
        
        # Extract sentiment by split
        trend_data = {}
        
        for split_name, results_df in self.sentiment_results.items():
            trend_data[split_name] = {
                'avg_sentiment': results_df['sentiment_score'].mean(),
                'positive_pct': (results_df['sentiment_label'] == 'positive').mean() * 100,
                'neutral_pct': (results_df['sentiment_label'] == 'neutral').mean() * 100,
                'negative_pct': (results_df['sentiment_label'] == 'negative').mean() * 100,
                'count': len(results_df),
                'split_num': int(split_name.split()[1])
            }
        
        # Convert to DataFrame
        trend_df = pd.DataFrame.from_dict(trend_data, orient='index')
        
        # Sort by split number
        trend_df = trend_df.sort_values('split_num')
        
        return trend_df
    
    def plot_timeline(self):
        """Plot sentiment timeline from all available data"""
        if not self.sentiment_results:
            print("No analyzed data available. Run analyze_all_splits() first.")
            return None
        
        # Combine all results
        all_tweets = []
        for df in self.sentiment_results.values():
            all_tweets.append(df)
        
        combined_df = pd.concat(all_tweets)
        
        # Ensure created_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(combined_df['created_at']):
            combined_df['created_at'] = pd.to_datetime(combined_df['created_at'], errors='coerce')
        
        # Group by day
        combined_df['date'] = combined_df['created_at'].dt.date
        daily_data = combined_df.groupby('date')['sentiment_score'].agg(['mean', 'count', 'std']).reset_index()
        daily_data.columns = ['date', 'avg_sentiment', 'tweet_count', 'sentiment_std']
        
        # Create the plot
        fig = px.line(
            daily_data,
            x='date',
            y='avg_sentiment',
            error_y=daily_data['sentiment_std'],
            labels={'date': 'Date', 'avg_sentiment': 'Average Sentiment'},
            title='Daily Sentiment Trend for Trump\'s Fiscal Policy'
        )
        
        # Customize the chart
        fig.update_traces(
            line=dict(width=3, color='#2980b9'),
            marker=dict(size=8, color='#2980b9'),
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.4f}<br>Tweets: %{customdata}<br>',
            customdata=daily_data['tweet_count']
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type='line',
            x0=daily_data['date'].min(),
            y0=0,
            x1=daily_data['date'].max(),
            y1=0,
            line=dict(color='gray', width=1, dash='dash')
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Sentiment Score (-1 to 1)',
            yaxis=dict(range=[-1, 1]),
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_period_comparison(self):
        """Plot sentiment comparison across time periods"""
        trend_df = self.get_trend_data()
        
        if trend_df is None or trend_df.empty:
            print("No trend data available.")
            return None
        
        # Add human-readable labels
        trend_df['period_label'] = trend_df.index.map(
            lambda x: self.date_splits[x]['name']
        )
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for average sentiment
        fig.add_trace(
            go.Bar(
                x=trend_df['period_label'],
                y=trend_df['avg_sentiment'],
                name='Average Sentiment',
                marker_color=trend_df['avg_sentiment'].apply(
                    lambda x: '#27ae60' if x > 0.1 else '#e74c3c' if x < -0.1 else '#95a5a6'
                ),
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.4f}<br>'
            ),
            secondary_y=False
        )
        
        # Add line chart for tweet count
        fig.add_trace(
            go.Scatter(
                x=trend_df['period_label'],
                y=trend_df['count'],
                name='Tweet Count',
                mode='lines+markers',
                marker=dict(color='#e74c3c'),
                line=dict(color='#e74c3c', width=2),
                hovertemplate='<b>%{x}</b><br>Tweets: %{y}<br>'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title='Sentiment Analysis by Time Period',
            xaxis=dict(
                title='Time Period',
                tickangle=-45
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=60, t=80, b=80),
            template='plotly_white',
            height=500
        )
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Average Sentiment", range=[-1, 1], secondary_y=False)
        fig.update_yaxes(title_text="Tweet Count", secondary_y=True)
        
        return fig
    
    def predict_long_term_impact(self):
        """Predict long-term sentiment based on historical pattern matching"""
        trend_df = self.get_trend_data()
        
        if trend_df is None or len(trend_df) < 2:
            print("Insufficient data for prediction. Need at least 2 time periods.")
            return None
        
        # Get initial and short-term sentiment values
        trend_df = trend_df.sort_values('split_num')
        initial_sentiment = trend_df.iloc[0]['avg_sentiment']
        short_term_sentiment = trend_df.iloc[1]['avg_sentiment']
        
        # Calculate trend direction
        trend_direction = short_term_sentiment - initial_sentiment
        
        # Calculate similarity to historical patterns
        similarities = {}
        
        for policy_name, policy_data in self.reference_policies.items():
            # Get historical values
            historical_initial = policy_data['initial_sentiment']
            historical_short_term = policy_data['short_term_sentiment']
            historical_long_term = policy_data['long_term_sentiment']
            
            # Calculate trend similarity
            historical_trend = historical_short_term - historical_initial
            trend_similarity = 1 / (1 + abs(trend_direction - historical_trend))
            
            # Calculate absolute value similarity
            absolute_similarity = 1 / (1 + abs(initial_sentiment - historical_initial) + 
                                     abs(short_term_sentiment - historical_short_term))
            
            # Combined similarity (weighted)
            combined_similarity = 0.7 * trend_similarity + 0.3 * absolute_similarity
            
            # Store results
            similarities[policy_name] = {
                'similarity': combined_similarity,
                'trend_similarity': trend_similarity,
                'absolute_similarity': absolute_similarity,
                'pattern': policy_data['pattern'],
                'historical_initial': historical_initial,
                'historical_short_term': historical_short_term,
                'historical_long_term': historical_long_term
            }
        
        # Sort by similarity
        sorted_similarities = dict(sorted(similarities.items(), 
                                        key=lambda x: x[1]['similarity'], 
                                        reverse=True))
        
        # Get top matches
        top_matches = list(sorted_similarities.items())[:2]
        
        # Calculate weighted prediction
        weights = [policy_data['similarity'] for _, policy_data in top_matches]
        adjustments = [policy_data['historical_long_term'] - policy_data['historical_short_term'] 
                     for _, policy_data in top_matches]
        
        # Normalize weights
        sum_weights = sum(weights)
        if sum_weights == 0:
            norm_weights = [1/len(weights)] * len(weights)
        else:
            norm_weights = [w/sum_weights for w in weights]
        
        # Calculate weighted adjustment
        weighted_adjustment = sum(adj * weight for adj, weight in zip(adjustments, norm_weights))
        
        # Apply adjustment to get prediction
        predicted_long_term = short_term_sentiment + weighted_adjustment
        
        # Cap within reasonable range
        predicted_long_term = max(-1.0, min(1.0, predicted_long_term))
        
        # Calculate confidence based on similarity
        confidence = min(0.9, top_matches[0][1]['similarity'])
        
        # Return prediction data
        return {
            'current_initial': initial_sentiment,
            'current_short_term': short_term_sentiment,
            'predicted_long_term': predicted_long_term,
            'confidence': confidence,
            'trend_direction': trend_direction,
            'top_matches': top_matches
        }
    
    def plot_historical_comparison(self):
        """Plot comparison with historical fiscal policy patterns"""
        prediction = self.predict_long_term_impact()
        
        if prediction is None:
            print("Cannot create historical comparison. Run predict_long_term_impact() first.")
            return None
        
        # Prepare data for visualization
        current_data = pd.DataFrame({
            'policy': ['Current Fiscal Policy 2025'],
            'timeframe': ['Initial', 'Short-term', 'Long-term (Predicted)'],
            'value': [
                prediction['current_initial'],
                prediction['current_short_term'],
                prediction['predicted_long_term']
            ],
            'type': ['Actual', 'Actual', 'Predicted']
        })
        
        # Add top 2 historical policies
        historical_data = []
        
        for policy_name, policy_data in prediction['top_matches']:
            historical_data.append(pd.DataFrame({
                'policy': [policy_name] * 3,
                'timeframe': ['Initial', 'Short-term', 'Long-term'],
                'value': [
                    policy_data['historical_initial'],
                    policy_data['historical_short_term'],
                    policy_data['historical_long_term']
                ],
                'type': ['Historical'] * 3
            }))
        
        # Combine data
        plot_data = pd.concat([current_data] + historical_data, ignore_index=True)
        
        # Create the plot
        fig = px.line(
            plot_data, 
            x='timeframe', 
            y='value', 
            color='policy',
            line_dash='type',
            markers=True,
            title='Current Fiscal Policy Compared to Most Similar Historical Patterns',
            labels={
                'timeframe': 'Time Frame',
                'value': 'Sentiment Score',
                'policy': 'Policy'
            },
            category_orders={
                'timeframe': ['Initial', 'Short-term', 'Long-term (Predicted)', 'Long-term']
            }
        )
        
        # Customize lines
        fig.update_traces(
            line=dict(width=4),
            marker=dict(size=10),
            selector=dict(name='Current Fiscal Policy 2025')
        )
        
        # Add confidence interval for prediction
        confidence = prediction['confidence']
        predicted_value = prediction['predicted_long_term']
        confidence_range = 0.2 * (1 - confidence)
        
        fig.add_shape(
            type="rect",
            x0=2.9,  # Just before Long-term
            x1=3.1,  # Just after Long-term
            y0=predicted_value - confidence_range,
            y1=predicted_value + confidence_range,
            line=dict(width=0),
            fillcolor="rgba(52, 152, 219, 0.2)"
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title='Time Frame',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1]),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def analyze_keywords(self):
        """Analyze keywords in the Twitter data"""
        if not self.sentiment_results:
            print("No analyzed data available. Run analyze_all_splits() first.")
            return None
        
        # Combine all data
        all_tweets = pd.concat(self.sentiment_results.values())
        
        # Extract keywords
        keywords = {}
        
        # Process each tweet
        for _, row in all_tweets.iterrows():
            if not isinstance(row['clean_text'], str) or pd.isna(row['clean_text']):
                continue
                
            words = row['clean_text'].lower().split()
            
            # Filter with stopwords
            stop_words = set(stopwords.words('english'))
            
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    if word not in keywords:
                        keywords[word] = {
                            'count': 0,
                            'sentiment_sum': 0,
                            'positive_count': 0,
                            'negative_count': 0,
                            'neutral_count': 0
                        }
                    
                    keywords[word]['count'] += 1
                    keywords[word]['sentiment_sum'] += row['sentiment_score']
                    
                    if row['sentiment_label'] == 'positive':
                        keywords[word]['positive_count'] += 1
                    elif row['sentiment_label'] == 'negative':
                        keywords[word]['negative_count'] += 1
                    else:
                        keywords[word]['neutral_count'] += 1
        
        # Calculate averages
        for word, stats in keywords.items():
            if stats['count'] > 0:
                stats['avg_sentiment'] = stats['sentiment_sum'] / stats['count']
                stats['positive_pct'] = (stats['positive_count'] / stats['count']) * 100
                stats['negative_pct'] = (stats['negative_count'] / stats['count']) * 100
                stats['neutral_pct'] = (stats['neutral_count'] / stats['count']) * 100
        
        # Convert to DataFrame
        keywords_df = pd.DataFrame.from_dict(keywords, orient='index')
        
        # Filter to keywords with at least 5 occurrences
        keywords_df = keywords_df[keywords_df['count'] >= 5]
        
        return keywords_df
    
    def plot_keyword_analysis(self):
        """Plot keyword sentiment analysis"""
        keywords_df = self.analyze_keywords()
        
        if keywords_df is None or keywords_df.empty:
            print("No keyword data available.")
            return None
        
        # Sort by absolute sentiment
        keywords_df['abs_sentiment'] = keywords_df['avg_sentiment'].abs()
        
        # Get top positive and negative keywords
        top_positive = keywords_df.sort_values('avg_sentiment', ascending=False).head(10)
        top_negative = keywords_df.sort_values('avg_sentiment').head(10)
        
        # Combine for visualization
        top_keywords = pd.concat([top_positive, top_negative])
        
        # Create the plot
        fig = px.bar(
            top_keywords.sort_values('avg_sentiment'),
            x=top_keywords.index,
            y='avg_sentiment',
            color='avg_sentiment',
            color_continuous_scale='RdBu_r',
            labels={
                'index': 'Keyword',
                'avg_sentiment': 'Average Sentiment',
                'count': 'Occurrence Count'
            },
            title='Top Positive and Negative Keywords in Fiscal Policy Discussion',
            hover_data=['count', 'positive_pct', 'negative_pct']
        )
        
        # Customize the chart
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Sentiment: %{y:.4f}<br>Count: %{customdata[0]}<br>Positive: %{customdata[1]:.1f}%<br>Negative: %{customdata[2]:.1f}%<br>',
            marker_line_width=0
        )
        
        # Add a horizontal line at y=0
        fig.add_shape(
            type='line',
            x0=-0.5,
            y0=0,
            x1=len(top_keywords)-0.5,
            y1=0,
            line=dict(color='gray', width=1, dash='dash')
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title='Keyword',
            yaxis_title='Average Sentiment',
            xaxis_tickangle=-45,
            yaxis=dict(range=[-1, 1]),
            coloraxis_showscale=False,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def export_data(self, filename='twitter_sentiment_export.json'):
        """Export analyzed data for use in external dashboards"""
        if not self.sentiment_results:
            print("No analyzed data available. Run analyze_all_splits() first.")
            return None
        
        export_data = {
            'date_splits': {k: {'name': v['name'], 'range': [str(r) for r in v['range']]} 
                          for k, v in self.date_splits.items()},
            'reference_policies': self.reference_policies,
            'trend_data': self.get_trend_data().to_dict(orient='records') if self.get_trend_data() is not None else None,
            'prediction': self.predict_long_term_impact(),
            'splits_summary': {}
        }
        
        # Add summary for each split
        for split_name, df in self.sentiment_results.items():
            # Calculate stats
            stats = self._calculate_stats(df)
            
            # Get top keywords for this split
            split_keywords = {}
            if 'clean_text' in df.columns:
                # Extract keywords
                keywords = {}
                
                # Process each tweet
                for _, row in df.iterrows():
                    if not isinstance(row['clean_text'], str) or pd.isna(row['clean_text']):
                        continue
                        
                    words = row['clean_text'].lower().split()
                    
                    # Filter with stopwords
                    stop_words = set(stopwords.words('english'))
                    
                    for word in words:
                        if len(word) > 3 and word not in stop_words:
                            if word not in keywords:
                                keywords[word] = {
                                    'count': 0,
                                    'sentiment_sum': 0
                                }
                            
                            keywords[word]['count'] += 1
                            keywords[word]['sentiment_sum'] += row['sentiment_score']
                
                # Calculate averages
                for word, word_stats in keywords.items():
                    if word_stats['count'] > 5:  # Only include words with at least 5 occurrences
                        split_keywords[word] = {
                            'count': word_stats['count'],
                            'avg_sentiment': word_stats['sentiment_sum'] / word_stats['count']
                        }
            
            # Store in export data
            export_data['splits_summary'][split_name] = {
                'stats': stats,
                'top_keywords': split_keywords
            }
        
        # Save to file
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            print(f"Data successfully exported to {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return None
    
    def create_prediction_card(self):
        """Create an interactive card displaying sentiment prediction"""
        prediction = self.predict_long_term_impact()
        
        if prediction is None:
            return HTML("""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #2980b9; margin-bottom: 15px;">Long-term Sentiment Prediction</h3>
                <p style="font-size: 16px;">Not enough data for prediction. Please upload and analyze data for at least 2 time periods.</p>
            </div>
            """)
        
        # Format confidence level
        confidence_pct = int(prediction['confidence'] * 100)
        confidence_text = "High" if confidence_pct >= 70 else "Medium" if confidence_pct >= 40 else "Low"
        
        # Format sentiment color
        sentiment_value = prediction['predicted_long_term']
        sentiment_color = "#27ae60" if sentiment_value > 0.1 else "#e74c3c" if sentiment_value < -0.1 else "#7f8c8d"
        
        # Format trend direction
        trend_direction = prediction['trend_direction']
        trend_icon = "↗️" if trend_direction > 0.05 else "↘️" if trend_direction < -0.05 else "→"
        
        # Generate HTML for similar policies
        similar_policies_html = ""
        for i, (policy_name, policy_data) in enumerate(prediction['top_matches'][:3]):
            similarity_pct = int(policy_data['similarity'] * 100)
            similar_policies_html += f"""
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <div style="width: 150px; font-weight: 500;">{i+1}. {policy_name}</div>
                <div style="width: 100px; height: 8px; background-color: #ecf0f1; border-radius: 4px; margin: 0 10px; overflow: hidden;">
                    <div style="height: 100%; width: {similarity_pct}%; background-color: #3498db;"></div>
                </div>
                <div>{policy_data['pattern']} ({similarity_pct}%)</div>
            </div>
            """
        
        # Create HTML output
        html_output = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #2980b9; margin-bottom: 15px;">Long-term Sentiment Prediction</h3>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                <div style="text-align: center; padding: 10px; background-color: white; border-radius: 8px; width: 30%;">
                    <div style="font-size: 14px; color: #7f8c8d;">Initial</div>
                    <div style="font-size: 24px; font-weight: 600; margin: 5px 0;">{prediction['current_initial']:.2f}</div>
                </div>
                <div style="text-align: center; padding: 10px; background-color: white; border-radius: 8px; width: 30%;">
                    <div style="font-size: 14px; color: #7f8c8d;">Short-term</div>
                    <div style="font-size: 24px; font-weight: 600; margin: 5px 0;">{prediction['current_short_term']:.2f}</div>
                </div>
                <div style="text-align: center; padding: 10px; background-color: white; border-radius: 8px; width: 30%;">
                    <div style="font-size: 14px; color: #7f8c8d;">Predicted Long-term</div>
                    <div style="font-size: 24px; font-weight: 600; margin: 5px 0; color: {sentiment_color};">{prediction['predicted_long_term']:.2f}</div>
                </div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <div style="font-weight: 600; margin-bottom: 5px;">Current Trend: {trend_icon} {trend_direction:.2f}</div>
                <div style="font-size: 14px; color: #7f8c8d;">Confidence: {confidence_text} ({confidence_pct}%)</div>
            </div>
            
            <div style="margin-top: 15px;">
                <div style="font-weight: 600; margin-bottom: 10px;">Most Similar Historical Patterns:</div>
                {similar_policies_html}
            </div>
        </div>
        """
        
        return HTML(html_output)

# Usage example
if __name__ == "__main__":
    analyzer = TwitterSentimentAnalyzer(load_model=False)
    analyzer.bulk_upload()
    analyzer.analyze_all_splits()
    
    # For interactive use, uncomment these lines:
    # prediction = analyzer.predict_long_term_impact()
    # analyzer.plot_timeline().show()
    # analyzer.plot_historical_comparison().show()
    
    # Export data for dashboard
    analyzer.export_data('twitter_sentiment_export.json')