import re
from datetime import datetime
from collections import Counter


def extract_recommendation(text: str) -> str:
    """Extract BUY/SELL/HOLD from agent output. Lenient matching."""
    # Look for the structured format first: **Rating:** BUY
    match = re.search(r'\*?\*?Rating\*?\*?:\s*\*?\*?(BUY|SELL|HOLD)\*?\*?', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: find standalone BUY/SELL/HOLD
    match = re.search(r'\b(BUY|SELL|HOLD)\b', text)
    if match:
        return match.group(1).upper()
    return "N/A"


def extract_conviction(text: str) -> float:
    """Extract conviction score (1-10) from agent output. Lenient matching."""
    # Structured format: **Conviction:** 8
    match = re.search(r'\*?\*?Conviction\*?\*?:\s*\*?\*?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    # Fallback patterns: "conviction (8/10)", "conviction: 8/10", "8 out of 10"
    match = re.search(r'conviction[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?', text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', text)
    if match:
        return float(match.group(1))
    return 0.0


def extract_section(text: str, header: str) -> str:
    """Extract a markdown section by header name."""
    pattern = rf'##\s*{re.escape(header)}\s*\n(.*?)(?=\n##\s|\Z)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


AGENT_DISPLAY_NAMES = {
    'Fundamental_Analyst': 'Fundamental Analysis',
    'Valuation_Analyst': 'Valuation Analysis',
    'Sentiment_Analyst': 'Sentiment Analysis',
}


def generate_report(debate_result: dict) -> dict:
    """Parse agent messages and build a structured report dict."""
    ticker = debate_result['ticker']
    yahoo_data = debate_result['yahoo_data']
    agent_messages = debate_result['agent_messages']

    report = {
        'ticker': ticker,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'current_price': float(yahoo_data['price']['close']),
        'day_high': float(yahoo_data['price']['high']),
        'day_low': float(yahoo_data['price']['low']),
        'agents': {},
        'consensus': None,
    }

    recommendations = []
    convictions = []

    for agent_name, messages in agent_messages.items():
        # Use the last message (second round / final position)
        last_msg = messages[-1] if messages else ''
        rec = extract_recommendation(last_msg)
        conv = extract_conviction(last_msg)

        report['agents'][agent_name] = {
            'display_name': AGENT_DISPLAY_NAMES.get(agent_name, agent_name),
            'raw': last_msg,
            'recommendation': rec,
            'conviction': conv,
            'data_summary': extract_section(last_msg, 'Data Summary'),
            'analysis': extract_section(last_msg, 'Analysis'),
            'risks': extract_section(last_msg, 'Key Risks'),
        }

        if rec != "N/A":
            recommendations.append(rec)
        if conv > 0:
            convictions.append(conv)

    # Derive consensus
    if recommendations:
        counts = Counter(recommendations)
        consensus_rec = counts.most_common(1)[0][0]
        agreement = counts.most_common(1)[0][1]
    else:
        consensus_rec = "N/A"
        agreement = 0

    avg_conviction = sum(convictions) / len(convictions) if convictions else 0.0

    report['consensus'] = {
        'recommendation': consensus_rec,
        'avg_conviction': avg_conviction,
        'agreement': agreement,
        'total_agents': len(agent_messages),
    }

    return report
