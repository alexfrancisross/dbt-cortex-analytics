
  
    

create or replace transient table DBT_CORTEX_LLMS.ANALYTICS.sentiment_analysis
    

    
    as (-- Sentiment analysis model that combines sentiment from all sources
-- This model aggregates sentiment scores from interactions, reviews, and tickets
-- Provides a unified view of customer sentiment across all touchpoints

WITH interaction_sentiment AS (
    SELECT
        customer_id,
        interaction_date,
        sentiment_score,
        'interaction' AS source_type
    FROM DBT_CORTEX_LLMS.ANALYTICS.fact_customer_interactions
),
review_sentiment AS (
    SELECT
        customer_id,
        review_date AS interaction_date,
        sentiment_score,
        'review' AS source_type
    FROM DBT_CORTEX_LLMS.ANALYTICS.fact_product_reviews
),
ticket_sentiment AS (
    SELECT
        customer_id,
        ticket_date AS interaction_date,
        sentiment_score,
        'ticket' AS source_type
    FROM DBT_CORTEX_LLMS.ANALYTICS.fact_support_tickets
)
SELECT * FROM interaction_sentiment
UNION ALL
SELECT * FROM review_sentiment
UNION ALL
SELECT * FROM ticket_sentiment
    )
;


  