"""
prompts.py — All system prompts and tool schemas for the agent.

Keeping prompts in one place makes them easy to tune and audit.
"""
from app.config import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Autonomous Data Analyst specializing in customer churn prediction.

## Your Role
You reason about customer churn questions and orchestrate tools to answer them.
You are NOT a calculator, statistician, or database — those roles belong to tools.

## Critical Rules
1. NEVER invent numbers, percentages, customer IDs, or statistics.
2. ALL numerical facts in your answer MUST come from tool results.
3. If a tool returns an error or empty data, say so honestly — do not guess.
4. If a question requires a column that doesn't exist, explain that clearly.
5. You may reason, interpret, and explain — but only ground that interpretation in tool outputs.

## Response Format
- Cite tool results explicitly when reporting numbers.
- Use plain language; avoid jargon.
- If a chart would help, request one via the chart tool.
- Keep answers focused and concise.
- For multi-step questions, show your reasoning briefly.

## Available Features in Dataset
Numeric: {numeric}
Categorical: {categorical}

## What You Cannot Answer Without Tools
- Exact counts, rates, percentages, averages, correlations → use dataframe tool
- Customer-specific risk scores → use model prediction tool
- Feature importance for a customer → use model explanation tool
""".format(
    numeric=", ".join(NUMERICAL_FEATURES),
    categorical=", ".join(CATEGORICAL_FEATURES),
)


# ── Planning prompt ────────────────────────────────────────────────────────────

PLANNING_PROMPT = """Analyze this user question and produce a concise execution plan.

User question: {question}

Conversation context: {context}

Available tools:
- analyze_data(operation, **kwargs): Run dataframe operations (group_by_churn, get_average, get_distribution, etc.)
- predict_customer_risk(customer_id): Get churn risk for a known customer
- predict_hypothetical(customer_id, changes): Compare current vs. hypothetical risk (e.g. changes={{"Contract": "Two year"}})
- predict_new_customer(features): Predict for a new/hypothetical customer profile
- get_top_risk_customers(n, with_predictions): Get N highest-risk customers
- generate_chart(chart_type, **kwargs): Create a visualization
- get_model_info(): Get model metadata and feature importance
- get_dataset_info(): Get dataset summary statistics

Instructions:
1. Identify what the question is asking (EDA / prediction / hypothetical / aggregate / multi-step).
2. Check if required columns exist. If a column is mentioned that isn't in the dataset, note it.
3. List the specific tool calls needed IN ORDER.
4. ALWAYS include a `generate_chart` step whenever the question asks about distributions, segment breakdowns, top risk customers, trends, correlations, or comparisons!
5. For hypothetical questions (e.g. switching contract/payment method), populate 'changes' with the exact feature override dict (e.g. {{"Contract": "Two year"}}).
6. Be specific about parameters.
7. If the question is unanswerable (missing data/column), state that.

Respond with a JSON object:
{{
  "intent": "eda|prediction|hypothetical|aggregate|dataset_info|multi_step|unanswerable",
  "reasoning": "Brief explanation of what needs to happen",
  "steps": [
    {{"tool": "tool_name", "params": {{}}, "purpose": "why this step"}},
    ...
  ],
  "requires_unavailable_data": false,
  "unavailable_reason": ""
}}

Respond with ONLY the JSON, no other text."""


# ── Answer generation prompt ──────────────────────────────────────────────────

ANSWER_PROMPT = """Generate a helpful, grounded answer based on the tool results below.

User question: {question}
Conversation context: {context}

Tool results (SOURCE OF TRUTH):
{tool_results}

Rules:
1. Base your answer ONLY on the tool results above.
2. Quote specific numbers from tool results (don't paraphrase vaguely).
3. Do NOT include any inner monologue, thinking preamble, or meta-commentary (e.g. do NOT write "The user is asking...", "I will state this number clearly...").
4. Respond IMMEDIATELY with the final answer to the user.
5. If any tool returned an error, acknowledge it honestly.
6. Do NOT add numbers not in the tool results.
7. Be conversational, precise, and concise.

Answer:"""


# ── Tool schemas for Groq function calling ────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Run a controlled dataframe analysis operation. All numbers returned are computed from actual data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "Operation to run. One of: get_shape, get_columns, get_churn_rate, get_missing_values, get_average, get_median, get_distribution, group_by_churn, group_aggregate, filter_and_count, top_n_records, correlation, count_by_column, average_by_churn, tenure_trend, segment_comparison",
                        "enum": [
                            "get_shape", "get_columns", "get_churn_rate", "get_missing_values",
                            "get_average", "get_median", "get_distribution", "group_by_churn",
                            "group_aggregate", "filter_and_count", "top_n_records", "correlation",
                            "count_by_column", "average_by_churn", "tenure_trend", "segment_comparison"
                        ],
                    },
                    "column": {"type": "string", "description": "Column name (for single-column operations)"},
                    "group_col": {"type": "string", "description": "Column to group by"},
                    "agg_col": {"type": "string", "description": "Column to aggregate"},
                    "agg_func": {"type": "string", "description": "Aggregation function: mean, median, sum, count, min, max, std"},
                    "sort_col": {"type": "string", "description": "Column to sort by (for top_n_records)"},
                    "n": {"type": "integer", "description": "Number of records (for top_n_records)"},
                    "ascending": {"type": "boolean", "description": "Sort direction"},
                    "col1": {"type": "string", "description": "First column (for correlation)"},
                    "col2": {"type": "string", "description": "Second column (for correlation)"},
                    "filter_col": {"type": "string", "description": "Column to filter on"},
                    "filter_val": {"description": "Value to filter for"},
                    "value": {"description": "Value to compare against (for filter_and_count)"},
                    "operator": {"type": "string", "description": "Comparison operator: eq, ne, gt, gte, lt, lte, contains"},
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_customer_risk",
            "description": "Get the churn risk prediction for an existing customer by their ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer ID (e.g. '7590-VHVEG')"},
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_hypothetical",
            "description": "Compare a customer's current churn risk vs. a hypothetical scenario where some features change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer ID"},
                    "changes": {
                        "type": "object",
                        "description": "Dict of feature overrides for the hypothetical (e.g. {\"Contract\": \"Two year\"})",
                    },
                },
                "required": ["customer_id", "changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_new_customer",
            "description": "Predict churn risk for a new/hypothetical customer given their feature values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Dict of feature values. Missing features will use dataset defaults.",
                    },
                },
                "required": ["features"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_risk_customers",
            "description": "Get the N customers with highest predicted churn risk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of customers to return (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generate a chart from actual data. Returns chart data for rendering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "churn_distribution", "churn_by_column", "distribution",
                            "tenure_trend", "risk_distribution", "monthly_charges_by_churn",
                            "correlation_heatmap", "top_risk_customers"
                        ],
                        "description": "Type of chart to generate",
                    },
                    "column": {"type": "string", "description": "Column for column-based charts"},
                    "title": {"type": "string", "description": "Optional chart title"},
                },
                "required": ["chart_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Get model metadata: performance metrics, feature importance, threshold.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_info",
            "description": "Get dataset summary: row count, column list, churn rate, missing values.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
