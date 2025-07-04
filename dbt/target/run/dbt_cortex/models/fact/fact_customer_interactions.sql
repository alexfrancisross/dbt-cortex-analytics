
  
    

create or replace transient table DBT_CORTEX_LLMS.ANALYTICS.fact_customer_interactions
    

    
    as (-- Fact model for customer interactions with sentiment analysis
-- This model enriches customer interactions with AI-generated sentiment scores
-- Uses Snowflake Cortex for sentiment analysis

SELECT
    i.interaction_id,
    i.customer_id,
    i.interaction_date,
    i.agent_id,
    i.interaction_type,
    i.interaction_notes,
    -- Add sentiment analysis using Snowflake Cortex
    SNOWFLAKE.CORTEX.SENTIMENT(i.interaction_notes) AS sentiment_score
FROM DBT_CORTEX_LLMS.STAGE.stg_customer_interactions i
WHERE i.interaction_notes IS NOT NULL
    )
;


  