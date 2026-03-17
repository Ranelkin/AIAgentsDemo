import time
import streamlit as st
import atexit
from src.util.log_config import setup_logging
from src.graph import run_debate, create_agents
from src.report import generate_report
from src.util import extract_ticker
from .stream import chat_stream

logger = setup_logging('chat.streamlit')

def _shutdown():
    try:
        logger.info("Shutting down application")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

atexit.register(_shutdown)

REC_COLORS = {
    'BUY': '#28a745',
    'SELL': '#dc3545',
    'HOLD': '#fd7e14',
    'N/A': '#6c757d',
}


def _render_report(report: dict, debate_result: dict):
    """Render the structured stock analysis report."""
    ticker = report['ticker']
    consensus = report['consensus']
    rec = consensus['recommendation']
    color = REC_COLORS.get(rec, '#6c757d')

    # ── Header ──
    st.markdown(f"# {ticker} Stock Analysis Report")
    st.caption(f"Generated {report['timestamp']}")
    st.markdown("---")

    # ── Consensus Overview ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<span style='font-size:2.2em;font-weight:bold;color:{color}'>{rec}</span>"
            f"<br><span style='color:#888'>Consensus</span></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.metric("Current Price", f"${report['current_price']:.2f}")
    with col3:
        st.metric("Avg Conviction", f"{consensus['avg_conviction']:.1f} / 10")
    with col4:
        st.metric("Agent Agreement", f"{consensus['agreement']} / {consensus['total_agents']}")

    st.markdown("---")

    # ── Agent Analysis Tabs ──
    agent_order = ['Fundamental_Analyst', 'Valuation_Analyst', 'Sentiment_Analyst']
    tab_labels = [report['agents'][a]['display_name'] for a in agent_order if a in report['agents']]
    tabs = st.tabs(tab_labels)

    for tab, agent_name in zip(tabs, agent_order):
        if agent_name not in report['agents']:
            continue
        agent = report['agents'][agent_name]
        with tab:
            # Agent recommendation badge
            a_rec = agent['recommendation']
            a_color = REC_COLORS.get(a_rec, '#6c757d')
            a_conv = agent['conviction']

            mcol1, mcol2 = st.columns([1, 4])
            with mcol1:
                st.markdown(
                    f"<div style='text-align:center;padding:0.5em;border-radius:8px;"
                    f"background:{a_color}20;border:2px solid {a_color}'>"
                    f"<span style='font-size:1.4em;font-weight:bold;color:{a_color}'>{a_rec}</span>"
                    f"<br>Conviction: {a_conv:.0f}/10</div>",
                    unsafe_allow_html=True,
                )
            with mcol2:
                if agent['data_summary']:
                    st.markdown("### Data Summary")
                    st.markdown(agent['data_summary'])

            if agent['analysis']:
                st.markdown("### Analysis")
                st.markdown(agent['analysis'])

            if agent['risks']:
                st.markdown("### Key Risks")
                st.markdown(agent['risks'])

            # Full raw output in collapsible
            with st.expander("View Full Agent Output"):
                st.markdown(agent['raw'])

    st.markdown("---")

    # ── Debate Transcript (inspectability) ──
    with st.expander("Full Debate Transcript", expanded=False):
        for msg in debate_result['chat_history']:
            name = msg.get('name', 'System')
            content = msg.get('content', '')
            if content:
                st.markdown(f"**{name}:**")
                st.markdown(content)
                st.markdown("---")


def chat_interface():
    # Streamlit config
    st.set_page_config(page_title="AlphaAgents", layout="wide")
    st.title("AlphaAgents")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    with st.sidebar:
        st.markdown(
            """
            - **Fundamental Agent**
            - **Sentiment Agent**
            - **Valuation Agent**
            """
        )
        if st.button("End Session"):
            st.success("Session cleaned up")
            st.rerun()

        # Set previous session conversations in sidebar
        st.markdown('---')
        st.markdown('Conversation History')

        if st.session_state.conversation_history:
            for idx, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
                if st.button(f"{conv['ticker']} - {conv['timestamp']}", key=f"hist_{idx}"):
                    st.session_state.selected_conversation = conv
        else:
            st.info("No conversation history yet")

        st.markdown("---")

        if st.button('Clear history'):
            st.session_state.conversation_history = []
            st.rerun()


    ####################
    # Chat configuration
    ####################

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("What would you like to do?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ticker = extract_ticker(prompt)
        st.write_stream(chat_stream(f'Analyzing the following ticker: {ticker}'))
        with st.chat_message("assistant"):
            try:
                with st.spinner(f"Agents are analyzing {ticker}..."):
                    debate_result = run_debate(ticker, mode='debate')
                    report = generate_report(debate_result)

                _render_report(report, debate_result)

                final_answer = (
                    f"**{ticker} — {report['consensus']['recommendation']}** "
                    f"(Conviction: {report['consensus']['avg_conviction']:.1f}/10, "
                    f"Agreement: {report['consensus']['agreement']}/{report['consensus']['total_agents']})"
                )

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                st.error(f"Error analyzing {ticker}: {e}")
                final_answer = f"Error: {str(e)}"

        # Session saving
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer #type: ignore
        })
