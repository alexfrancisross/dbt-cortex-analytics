# Customer Intelligence Hub Quickstart

A comprehensive customer analytics platform that leverages Snowflake Cortex AI functions, dbt transformations, and interactive Streamlit dashboards to unlock deep insights from customer data across multiple touchpoints.

## 📊 Overview

The Customer Intelligence Hub demonstrates how AI-powered analytics can transform customer understanding across departments:

- **🧠 AI-Powered Insights**: Uses Snowflake Cortex LLM functions to extract sentiment, classify personas, and analyze unstructured text
- **🔄 Modern Data Stack**: Built with dbt for transformations, Snowflake for data warehousing, and Streamlit for visualization
- **🌐 Multi-language Support**: Processes customer feedback in multiple languages with automatic translation
- **📈 Cross-Department Value**: Provides actionable insights for Marketing, Sales, Support, Finance, and HR teams
- **🤖 Natural Language Querying**: Cortex Analyst integration for conversational data exploration

## 🏗️ Project Structure

```
dbt_cortex_analytics/
├── dbt/                          # dbt project for data transformations
│   ├── models/
│   │   ├── staging/             # Raw data cleaning and staging
│   │   │   ├── stg_customers.sql
│   │   │   ├── stg_customer_interactions.sql
│   │   │   ├── stg_product_reviews.sql
│   │   │   ├── stg_support_tickets.sql
│   │   │   ├── schema.yml
│   │   │   └── sources.yml
│   │   ├── fact/                # Core business entity tables
│   │   │   ├── fact_customer_interactions.sql
│   │   │   ├── fact_product_reviews.sql
│   │   │   ├── fact_support_tickets.sql
│   │   │   └── schema.yml
│   │   └── analysis/            # AI-enhanced analytics models
│   │       ├── customer_persona_signals.sql
│   │       ├── sentiment_analysis.sql
│   │       ├── sentiment_trends.sql
│   │       ├── ticket_patterns.sql
│   │       ├── insight_summaries.sql
│   │       └── schema.yml
│   ├── seeds/                   # Sample data files (JSON format)
│   │   ├── customer_interactions.json
│   │   ├── customers.json
│   │   ├── product_reviews.json
│   │   └── support_tickets.json
│   ├── macros/
│   │   └── generate_schema_name.sql
│   ├── setup.sql               # Snowflake environment setup script
│   └── dbt_project.yml         # dbt configuration
├── streamlit/                   # Interactive dashboard application
│   ├── src/
│   │   ├── components/         # Dashboard components (6 tabs)
│   │   │   ├── overview.py     # KPIs and sentiment trends
│   │   │   ├── segmentation.py # Customer persona analysis
│   │   │   ├── sentiment_experience.py # Cross-channel sentiment
│   │   │   ├── product_feedback.py # Review analysis
│   │   │   ├── support_ops.py  # Support ticket analytics
│   │   │   └── cortex_analyst.py # Natural language querying
│   │   ├── sql/               # SQL queries organized by component
│   │   ├── utils/             # Utility functions
│   │   │   ├── database.py    # Snowflake connections
│   │   │   ├── chart_factory.py # Visualization generation
│   │   │   ├── chart_intelligence.py # AI chart recommendations
│   │   │   ├── theme.py       # Dynamic theming
│   │   │   └── accessibility.py # WCAG compliance
│   │   ├── assets/            # Static files (CSS, logos)
│   │   ├── streamlit_app.py   # Main application
│   │   └── requirements.txt   # Python dependencies
│   ├── cortex_analyst/        # Cortex Analyst semantic model
│   │   ├── semantic_model.yaml
│   │   └── upload_semantic_model.sh
│   ├── deploy_streamlit_to_snowflake.sh
│   └── README.md
├── data/                       # Sample datasets and generator
│   ├── generator/
│   │   ├── generate_synthetic_data.py
│   │   └── readme-instructions.md
│   └── samples/               # Pre-generated sample datasets
│       ├── 10_Customers/
│       ├── 100_Customers/
│       └── 1000_Customers/
├── snowflake_sql/             # Pure SQL implementation
│   ├── snowflake_sql.sql
│   └── snowflake_sql_notebook.ipynb
├── img/                       # Documentation images
└── snowflake-dbt-cortex-llms-quickstart.md # Detailed setup guide
```

## 🎯 Key Features

### Customer Intelligence Dashboard
- **📊 Overview**: KPIs, sentiment trends, churn risk analysis, and interaction metrics
- **🎯 Segmentation**: Customer persona analysis, value segmentation, and behavioral clustering
- **💭 Sentiment Experience**: Cross-channel sentiment tracking, recovery metrics, and volatility analysis
- **📝 Product Feedback**: Multi-language review analysis with automatic translation and sentiment scoring
- **🎧 Support Operations**: Ticket analytics, channel effectiveness, and resolution metrics
- **🤖 Cortex Analyst**: Natural language querying with intelligent chart recommendations

### AI-Powered Analytics
- **Sentiment Analysis**: Emotional insights from customer interactions, reviews, and support tickets using `SNOWFLAKE.CORTEX.SENTIMENT`
- **Persona Classification**: Automatic customer segmentation based on behavior patterns using `SNOWFLAKE.CORTEX.CLASSIFY_TEXT`
- **Multilingual Processing**: Supports customer feedback in English, Spanish, French, German, Italian, and Portuguese using `SNOWFLAKE.CORTEX.TRANSLATE`
- **Predictive Insights**: Churn risk scoring and upsell opportunity identification
- **Natural Language Generation**: AI-generated summaries and insights using `SNOWFLAKE.CORTEX.COMPLETE`

### Data Pipeline Architecture
- **Staging Layer**: Clean and structure raw JSON data with proper typing and validation
- **Fact Layer**: Core business entities enriched with Cortex AI functions
- **Analytics Layer**: Advanced models with persona signals, sentiment trends, and predictive insights

## 🚀 Quick Start

### Prerequisites

- Snowflake account with Cortex AI functions enabled
- dbt Cloud account (or dbt Core installed locally)
- Python 3.8+ for Streamlit application
- Git for version control

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Snowflake-Labs/dbt_cortex_analytics.git
cd dbt_cortex_analytics

# Set up Python environment for Streamlit
conda create -n customer-analytics python=3.11
conda activate customer-analytics
pip install -r streamlit/src/requirements.txt

# Install dbt (if using locally)
pip install dbt-core dbt-snowflake
```

### 2. Snowflake Configuration

1. Create database and schemas by running the setup script:
```sql
-- Execute dbt/setup.sql in your Snowflake worksheet
-- This creates:
-- - Database: DBT_CORTEX_LLMS
-- - Schemas: STAGE, ANALYTICS, SEMANTIC_MODELS
-- - Warehouse: CORTEX_WH
-- - Role: DBT_ROLE with proper permissions
```

2. Configure dbt profiles (`~/.dbt/profiles.yml`):
```yaml
dbt_cortex:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: <your-account>
      user: <your-username>
      private_key_path: <path-to-private-key>
      database: DBT_CORTEX_LLMS
      warehouse: CORTEX_WH
      schema: ANALYTICS
      role: DBT_ROLE
```

### 3. Data Pipeline Execution

```bash
# Navigate to dbt directory
cd dbt

# Install dependencies and run models
dbt deps
dbt seed  # Load sample JSON data
dbt run   # Execute transformations with Cortex AI
dbt test  # Validate data quality

# Generate documentation
dbt docs generate
dbt docs serve
```

### 4. Deploy Streamlit Application

```bash
# Deploy to Snowflake (recommended)
cd streamlit
./deploy_streamlit_to_snowflake.sh

# Or run locally
cd src
# Configure .streamlit/secrets.toml with your credentials
streamlit run streamlit_app.py
```

## 📊 Data Model

### Source Data Structure
- **Customer Interactions**: Customer service notes with sentiment analysis
- **Product Reviews**: Multi-language product feedback with ratings and automatic translation
- **Support Tickets**: Customer support requests with AI-powered categorization and priority scoring
- **Customer Profiles**: Demographics with AI-derived persona classification and lifetime value

### dbt Model Layers

#### 1. Staging Models (`STAGE` schema)
- `stg_customers`: Cleaned customer profile data with persona classification
- `stg_customer_interactions`: Parsed interaction notes with metadata
- `stg_product_reviews`: Structured review data with language detection
- `stg_support_tickets`: Categorized support requests with status tracking

#### 2. Fact Models (`ANALYTICS` schema)
- `fact_customer_interactions`: Interactions enhanced with sentiment scores using Cortex AI
- `fact_product_reviews`: Reviews with sentiment analysis and English translations
- `fact_support_tickets`: Tickets with AI-powered priority classification and remedy extraction

#### 3. Analytics Models (`ANALYTICS` schema)
- `customer_persona_signals`: Comprehensive customer profiles with churn risk and upsell scoring
- `sentiment_analysis`: Cross-channel sentiment tracking and aggregation
- `sentiment_trends`: Temporal sentiment analysis with volatility metrics
- `ticket_patterns`: Support ticket trend analysis and categorization
- `insight_summaries`: AI-generated customer behavior summaries

## 🎨 Streamlit Dashboard Components

### 📈 Overview Dashboard
- Customer satisfaction KPIs and trends
- Sentiment distribution across channels
- Churn risk analysis and predictions
- Interaction volume and engagement metrics

### 🎯 Customer Segmentation
- AI-driven persona analysis and distribution
- Value segment performance metrics
- Churn vs. upsell opportunity matrix
- Behavioral clustering and insights

### 💭 Sentiment Experience
- Cross-channel sentiment alignment analysis
- Recovery rate tracking and trends
- Sentiment volatility and stability metrics
- Persona-specific sentiment patterns

### 📝 Product Feedback
- Multi-language review analysis with automatic translation
- Rating trends and distribution patterns
- Recent review highlights and sentiment correlation
- Language-specific feedback insights

### 🎧 Support Operations
- Ticket volume trends and seasonal patterns
- Channel effectiveness and resolution metrics
- Agent performance and customer effort scores
- Priority distribution and escalation analysis

### 🤖 Cortex Analyst
- Natural language querying interface
- AI-generated insights and recommendations
- Interactive data exploration with follow-up questions
- Intelligent chart recommendations based on query results

## 🛠️ Technical Architecture

### Data Stack
- **Data Warehouse**: Snowflake with Cortex AI functions for sentiment analysis, translation, and text classification
- **Transformation**: dbt for SQL-based transformations with AI enhancement
- **Visualization**: Streamlit with Plotly for interactive charts and natural language querying
- **AI/ML**: Snowflake Cortex for sentiment analysis, text processing, and persona classification

### Key Technologies
- **Snowflake Cortex Functions**: `SENTIMENT`, `TRANSLATE`, `CLASSIFY_TEXT`, `COMPLETE`, `SUMMARIZE`
- **dbt**: Data modeling, testing, documentation, and lineage tracking
- **Streamlit**: Interactive web application with real-time data updates
- **Plotly**: Advanced data visualizations with interactive features
- **Pandas**: Data manipulation and analysis for dashboard components

## 📚 Sample Data

The project includes synthetic customer data demonstrating realistic business scenarios:

### Dataset Characteristics
- **Customer Personas**: Satisfied, frustrated, neutral, and mixed behavior patterns
- **Multilingual Reviews**: Product feedback in 6 languages with automatic translation
- **Correlated Interactions**: Consistent sentiment patterns across all touchpoints
- **Support Scenarios**: Various ticket types, priorities, and resolution patterns

### Available Sample Sizes
- **10 Customers**: Quick testing and development (default)
- **100 Customers**: Medium-scale demonstrations
- **1000 Customers**: Large-scale performance testing

### Generating Custom Data

```bash
cd data/generator

# Generate default dataset (100 customers)
python generate_synthetic_data.py

# Generate larger dataset
python generate_synthetic_data.py --num-records 1000

# Generate with specific output directory
python generate_synthetic_data.py --output-dir ../samples/custom_data

# Use templates only (faster generation)
python generate_synthetic_data.py --use-templates-only
```

## 🔒 Security & Best Practices

### Authentication
- **Private Key Authentication**: RSA key pairs for secure Snowflake connections
- **PAT Token Support**: Personal Access Tokens for Cortex Analyst API
- **Role-Based Access**: Proper Snowflake role and privilege management

### Data Protection
- **Credential Management**: Secure storage of authentication tokens
- **Environment Variables**: Sensitive configuration excluded from version control
- **Connection Security**: Encrypted connections and secure data transfer

### Development Best Practices
- **Schema Validation**: Comprehensive dbt tests for data quality
- **Error Handling**: Graceful failure handling in Streamlit components
- **Performance Optimization**: Efficient SQL queries and caching strategies

## 📖 Documentation

### Available Resources
- **Quickstart Guide**: `snowflake-dbt-cortex-llms-quickstart.md` - Complete 70-minute tutorial
- **Streamlit Documentation**: `streamlit/README.md` - Dashboard setup and features
- **Data Generator Guide**: `data/generator/readme-instructions.md` - Custom data generation
- **dbt Documentation**: Generated via `dbt docs generate` - Model lineage and tests

### Visual Documentation
- **Architecture Diagram**: `img/erd.png` - Entity relationship diagram
- **Data Lineage**: `img/dbt-dag.png` - dbt transformation flow
- **Dashboard Screenshots**: `img/SiS-demo.gif` - Interactive demo

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper testing
4. Update documentation as needed
5. Submit a pull request with detailed description

### Development Guidelines
- Follow dbt naming conventions for models
- Add comprehensive tests for new models
- Update schema.yml files with proper documentation
- Test Streamlit components across different data scenarios

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- **Setup Issues**: Check the quickstart guide for detailed instructions
- **dbt Questions**: Review dbt documentation and model schemas
- **Streamlit Issues**: Check component documentation and debug mode
- **Snowflake Cortex**: Consult Snowflake Cortex documentation for AI function usage
- **Bug Reports**: Open GitHub issues with detailed reproduction steps

### Common Issues
- **Cortex AI Functions**: Ensure your Snowflake account has Cortex AI enabled
- **Authentication**: Verify private key setup and role permissions
- **Data Loading**: Check file paths and JSON format in seed files
- **Performance**: Use appropriate warehouse sizes for your data volume

---

**Powered by Snowflake ❄️ and dbt 🧱** 