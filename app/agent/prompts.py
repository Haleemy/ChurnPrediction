"""
prompts.py — All system prompts and tool schemas for the agent.

Keeping prompts in one place makes them easy to tune and audit.
"""
from app.config import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Autonomous Data Analyst specializing in customer churn prediction.

## Your Role
You reason about customer churn questions and present computed insights clearly, professionally, and accurately.

## Critical Rules
1. NEVER invent numbers, percentages, customer IDs, or statistics.
2. ALL numerical facts in your answer MUST come from tool results.
3. If a tool returns an error or empty data, state that clearly and concisely.
4. If a question requires a column that doesn't exist, explain that clearly.
5. Present answers directly, cleanly, and professionally.

## Response Style & Formatting
- Do NOT use meta phrases like "(as returned by the model)", "Source: tool_name", or "The tool returned...".
- Present data directly and authoritatively without robotic disclaimers.
- Use clean Markdown tables for list data. If the user asks for 10 items, list all 10 items in the table.
- Use clear bullet points and bold numbers.
- Keep answers focused, executive-ready, and concise.

## Available Features in Dataset
Numeric: {numeric}
Categorical: {categorical}
""".format(
    numeric=", ".join(NUMERICAL_FEATURES),
    categorical=", ".join(CATEGORICAL_FEATURES),
)


# ── Planning prompt ────────────────────────────────────────────────────────────

PLANNING_PROMPT = """Analyze this user question and produce a concise execution plan.

User question: {question}

Conversation context: {context}

Available tools:
- analyze_data(operation, **kwargs): Run dataframe operations.
  Supported operations:
  • "correlation": calculate correlation between col1 and col2 (or overall numeric correlation if col1/col2 omitted). Parameters: col1, col2.
  • "group_by_churn": churn rate breakdown by a categorical column. Parameters: column.
  • "average_by_churn": mean of numeric column for churned vs non-churned. Parameters: column.
  • "get_average": overall average of numeric column. Parameters: column.
  • "get_median": median of numeric column. Parameters: column.
  • "get_distribution": histogram/stats distribution of a column. Parameters: column.
  • "count_by_column": value counts for a column. Parameters: column.
  • "filter_and_count": count rows matching condition. Parameters: column, value, operator ("eq", "gt", "lt", etc.).
  • "group_aggregate": aggregate agg_col grouped by group_col using agg_func ("mean", "sum", "count", etc.).
  • "tenure_trend": churn rate by tenure range.
  • "segment_comparison": compare metrics across segments. Parameters: segment_col, metric_col.
  • "get_churn_rate": overall dataset churn rate.
  • "get_shape": total rows and columns.
  • "get_columns": column metadata list.
  • "get_missing_values": count missing values per column.
- predict_customer_risk(customer_id): Get churn risk for a known customer ID (e.g. '7590-VHVEG')
- predict_hypothetical(customer_id, changes): Compare current vs. hypothetical risk (e.g. changes={{"Contract": "Two year"}})
- predict_new_customer(features): Predict for a new customer profile
- get_top_risk_customers(n): Get N highest-risk customers
- generate_chart(chart_type, **kwargs): Create a Plotly visualization.
  Supported chart_types:
  • "correlation_heatmap": correlation matrix heatmap of numeric features
  • "feature_importance": bar chart of top model feature importances
  • "churn_distribution": donut chart of overall churn
  • "churn_by_column": bar chart of churn rate by column (requires column="ColumnName")
  • "monthly_charges_by_churn": box plot of monthly charges by churn
  • "tenure_trend": combo bar/line chart of churn by tenure range
  • "distribution": histogram of a column (requires column="ColumnName")
  • "risk_distribution": histogram of predicted risk scores
  • "top_risk_customers": horizontal bar chart of top N risk customers
- get_model_info(): Get ML model metadata and feature importance
- get_dataset_info(): Get dataset summary statistics (row count, column list, overall churn rate)

Instructions:
1. Identify what the question is asking (EDA / prediction / hypothetical / aggregate / model_info / correlation / multi-step).
2. Check if required columns exist.
3. List the specific tool calls needed IN ORDER.
4. ALWAYS include a `generate_chart` step whenever the question asks about distributions, segment breakdowns, top risk customers, trends, correlations, feature importances, or comparisons!
5. For correlation questions (e.g. "correlation between tenure and monthly charges"), use `analyze_data(operation="correlation", col1="tenure", col2="MonthlyCharges")` AND `generate_chart(chart_type="correlation_heatmap")`.
6. For model info / feature importance questions, use `get_model_info()` AND `generate_chart(chart_type="feature_importance")`.
7. For hypothetical questions, populate 'changes' dict with exact overrides (e.g. {{"Contract": "Two year"}}).
8. If the question is unanswerable (missing data/column), state that with intent="unanswerable".

Respond with a JSON object:
{{
  "intent": "eda|prediction|hypothetical|aggregate|dataset_info|model_info|multi_step|unanswerable",
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

ANSWER_PROMPT = """Generate a clean, executive-ready answer based on the tool results below.

User question: {question}
Conversation context: {context}

Tool results (SOURCE OF TRUTH):
{tool_results}

Rules:
1. Base your answer ONLY on the tool results above.
2. Quote specific numbers from tool results accurately.
3. Do NOT write meta-disclaimers or citations like "(as returned by the model)", "Source: tool_name", "The tool returned...", or inner thoughts.
4. If the user asks for N items (e.g. top 10), present all N items in a clear Markdown table.
5. Respond IMMEDIATELY with the final polished answer.
6. Be direct, executive-ready, and concise.

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
