from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor
from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from src.tools import retrieve_yahoo_data
from src.tools.yahoo.yahoo import format_valuation_context, format_sentiment_context
from src.tools.sec_filings import create_tenk_filing_repl, create_tenq_filing_repl
from src.util.log_config import setup_logging
from src.config import llm_config

logger = setup_logging('autogen_agents')

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
instrumator = AutogenAgentChatInstrumentor()
assert instrumator
instrumator.instrument(tracer_provider=tracer_provider)

STRUCTURED_OUTPUT_FORMAT = """
YOUR RESPONSE MUST follow this exact format:

## Data Summary
[List the specific data points you used, with exact numbers]

## Analysis
[Your interpretation of the data — cite specific numbers to support each claim]

## Recommendation
- **Rating:** [BUY / SELL / HOLD]
- **Conviction:** [1-10]
- **Rationale:** [One sentence summarizing why]

## Key Risks
- [Risk 1 — grounded in specific data]
- [Risk 2 — grounded in specific data]
"""

SECOND_ROUND_INSTRUCTIONS = """
SECOND ROUND: Review your peers' analyses. If you adjust your rating or conviction,
state exactly which data point from their analysis changed your view.
Maintain the same output format as above."""


def create_agents(ticker: str, yahoo_data: dict):
    """Create the three specialized agents for a given ticker"""

    valuation_context = format_valuation_context(yahoo_data)
    sentiment_context = format_sentiment_context(yahoo_data)

    fundamental_agent = AssistantAgent(
        name="Fundamental_Analyst",
        system_message=f"""You are a fundamental equity analyst for {ticker}. Your job is to
analyze the company's latest SEC filings (10-K and 10-Q) and provide a data-grounded
investment recommendation.

You have access to two tools:
- tenk_repl: Execute Python code against the latest 10-K filing (use the `filing` variable)
- tenq_repl: Execute Python code against the latest 10-Q filing (use the `filing` variable)

INSTRUCTIONS:
1. Use the tools to extract specific financial metrics: revenue, net income, EPS,
   total debt, cash & equivalents, operating cash flow, and gross margin.
2. Compare year-over-year or quarter-over-quarter trends where available.
3. Cite exact dollar amounts and percentages from the filings.
4. Do NOT make generic statements — every claim must reference a specific number.

{STRUCTURED_OUTPUT_FORMAT}

FIRST ROUND: Provide your independent analysis based solely on filing data.
{SECOND_ROUND_INSTRUCTIONS}""",
        llm_config=llm_config,
        function_map={
                    "tenk_repl": lambda code: create_tenk_filing_repl(ticker).run(code),
                    "tenq_repl": lambda code: create_tenq_filing_repl(ticker).run(code)
                }
    )

    valuation_agent = AssistantAgent(
        name="Valuation_Analyst",
        system_message=f"""You are a valuation equity analyst for {ticker}. Your job is to
analyze price trends, trading volume, and valuation patterns to provide a data-grounded
investment recommendation.

HERE IS YOUR DATA — use these exact numbers in your analysis:

{valuation_context}

INSTRUCTIONS:
1. Assess the current price relative to the 1-month and 1-year trends.
2. Identify whether the stock is near its 52-week high/low and what that implies.
3. Analyze volume trends for signs of accumulation or distribution.
4. Calculate key metrics: current price vs. 1-year average, distance from 52-week high/low.
5. Do NOT make generic statements — every claim must reference a specific number from the data above.

{STRUCTURED_OUTPUT_FORMAT}

FIRST ROUND: Provide your independent analysis based solely on price/volume data.
{SECOND_ROUND_INSTRUCTIONS}""",
        llm_config=llm_config
    )

    sentiment_agent = AssistantAgent(
        name="Sentiment_Analyst",
        system_message=f"""You are a sentiment equity analyst for {ticker}. Your job is to
analyze market sentiment, news flow, and analyst consensus to provide a data-grounded
investment recommendation.

HERE IS YOUR DATA — use these exact numbers in your analysis:

{sentiment_context}

INSTRUCTIONS:
1. Interpret the overall sentiment score and what it signals.
2. Highlight the most bullish and most bearish news headlines with their scores.
3. Compare the current price to analyst targets (low, mean, median, high).
4. Assess whether analyst consensus supports upside or downside.
5. Do NOT make generic statements — every claim must reference a specific number from the data above.

{STRUCTURED_OUTPUT_FORMAT}

FIRST ROUND: Provide your independent analysis based solely on sentiment/news data.
{SECOND_ROUND_INSTRUCTIONS}""",
        llm_config=llm_config
    )

    return [fundamental_agent, valuation_agent, sentiment_agent]


def run_debate(ticker: str, mode: str = "debate") -> dict:
    """
    Multi-agent analysis for a stock.
    Returns a structured dict with chat_history, per-agent messages, and metadata.
    """
    logger.info(f"Starting {mode} mode analysis for {ticker}")

    yahoo_data = retrieve_yahoo_data(ticker)
    agents = create_agents(ticker, yahoo_data)

    if mode == "debate":
        groupchat = GroupChat(
            agents=agents, #type: ignore
            messages=[],
            max_round=6,
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False
        )
        manager_system_message = """You are a helpful assistant skilled at coordinating a group
            of other agents to solve a task. You make sure that every agent in
            the group chat has a chance to speak at least twice. Each agent can
            not decide for the whole group. They are tasked with coming to a
            consensus. You must invoke all agents before deciding to Terminate.
            Reply "TERMINATE" at the end when everything is done."""
    else:  # collaboration
        groupchat = GroupChat(
            agents=agents, #type: ignore
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin"
        )
        manager_system_message = """You are a helpful assistant skilled at coordinating a group
            of other agents to solve a task. You make sure that every agent in
            the group chat has a chance to speak at least twice. When all agents
            provide their analysis, consolidate inputs of all agent into a report.
            Reply "TERMINATE" at the end when everything is done."""

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        system_message=manager_system_message
    )

    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    result = user_proxy.initiate_chat(
        recipient=manager,
        message=f"Analyze {ticker} and provide a consensus BUY/SELL/HOLD recommendation with reasoning.",
        clear_history=True,
        silent=False
    )

    # Parse chat history into per-agent messages
    agent_names = ['Fundamental_Analyst', 'Sentiment_Analyst', 'Valuation_Analyst']
    agent_messages = {name: [] for name in agent_names}
    for msg in result.chat_history:
        name = msg.get('name', '')
        if name in agent_messages:
            agent_messages[name].append(msg.get('content', ''))

    return {
        'chat_history': result.chat_history,
        'agent_messages': agent_messages,
        'yahoo_data': yahoo_data,
        'ticker': ticker,
        'summary': result.summary,
    }
